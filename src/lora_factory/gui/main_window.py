"""Connected single-window PySide6 shell for LoRA Factory."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lora_factory.config.models import ProjectConfig
from lora_factory.gui.completion_view import CompletionView
from lora_factory.gui.contracts import ApplicationController, PipelineUpdate, as_mapping
from lora_factory.gui.dataset_review import DatasetReviewView
from lora_factory.gui.progress_view import ProgressView
from lora_factory.gui.project_editor import ProjectEditor
from lora_factory.gui.settings_view import SettingsView
from lora_factory.gui.setup_view import SetupView
from lora_factory.gui.workers import ApplicationActionWorker, PipelineWorker


class MainWindow(QMainWindow):
    """Route GUI actions exclusively through an injected application controller."""

    pipeline_started = Signal(str)
    pipeline_completed = Signal(object)
    pipeline_failed = Signal(str)

    def __init__(self, controller: ApplicationController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._action_thread: QThread | None = None
        self._action_worker: ApplicationActionWorker | None = None
        self._active_config: ProjectConfig | None = None
        self._pending_refinement_project_id = ""
        self._recent_values: list[dict[str, Any]] = []
        self.setWindowTitle("LoRA Factory")
        self.resize(1240, 820)
        self.setMinimumSize(960, 640)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(300)
        sidebar.setStyleSheet(
            "QFrame#sidebar { background: #182230; color: white; }"
            "QFrame#sidebar QPushButton { text-align: left; padding: 8px; color: white; "
            "border: 0; } QFrame#sidebar QPushButton:hover { background: #344054; }"
            "QFrame#sidebar QLabel { color: white; }"
        )
        side_layout = QVBoxLayout(sidebar)
        brand = QLabel("LoRA Factory")
        brand.setStyleSheet("font-size: 21px; font-weight: 650; padding: 10px 4px;")
        side_layout.addWidget(brand)
        self.new_project_button = self._nav_button("New Project", "navNewProject")
        self.recent_button = self._nav_button("Recent Projects", "navRecent")
        self.running_button = self._nav_button("Running", "navRunning")
        self.completed_button = self._nav_button("Completed", "navCompleted")
        self.failed_button = self._nav_button("Failed / Resume", "navFailed")
        for button in (
            self.new_project_button,
            self.recent_button,
            self.running_button,
            self.completed_button,
            self.failed_button,
        ):
            side_layout.addWidget(button)
        recent_label = QLabel("Projects")
        recent_label.setStyleSheet("font-weight: 600; padding-top: 8px;")
        side_layout.addWidget(recent_label)
        self.recent_list = QListWidget()
        self.recent_list.setObjectName("recentProjects")
        self.recent_list.setStyleSheet("background: #101828; color: white; border: 0;")
        self.recent_list.itemActivated.connect(self._activate_recent)
        side_layout.addWidget(self.recent_list, 1)
        self.dataset_button = self._nav_button("Dataset Review", "navDataset")
        self.progress_button = self._nav_button("Progress", "navProgress")
        self.completion_button = self._nav_button("Completion", "navCompletion")
        self.setup_button = self._nav_button("Setup", "navSetup")
        self.settings_button = self._nav_button("Settings", "navSettings")
        for button in (
            self.dataset_button,
            self.progress_button,
            self.completion_button,
            self.setup_button,
            self.settings_button,
        ):
            side_layout.addWidget(button)
        root_layout.addWidget(sidebar)

        self.pages = QStackedWidget()
        self.pages.setObjectName("mainPages")
        self.setup_view = SetupView()
        self.project_editor = ProjectEditor()
        self.dataset_review = DatasetReviewView()
        self.progress_view = ProgressView()
        self.completion_view = CompletionView()
        self.settings_view = SettingsView()
        for page in (
            self.setup_view,
            self.project_editor,
            self.dataset_review,
            self.progress_view,
            self.completion_view,
            self.settings_view,
        ):
            self.pages.addWidget(page)
        root_layout.addWidget(self.pages, 1)
        self.setCentralWidget(root)

        self._connect_actions()
        self._load_destinations()
        self.refresh_recent_projects()
        self.refresh_gpus()
        self.refresh_setup(initial=True)

    def _connect_actions(self) -> None:
        self.new_project_button.clicked.connect(self._new_project)
        self.recent_button.clicked.connect(self.refresh_recent_projects)
        self.running_button.clicked.connect(
            lambda: self._filter_recent({"active", "running", "awaiting_review"})
        )
        self.completed_button.clicked.connect(lambda: self._filter_recent({"completed", "ready"}))
        self.failed_button.clicked.connect(
            lambda: self._filter_recent(
                {"failed", "failed_recoverable", "failed_fatal", "cancelled"}
            )
        )
        self.dataset_button.clicked.connect(
            lambda: self.pages.setCurrentWidget(self.dataset_review)
        )
        self.progress_button.clicked.connect(
            lambda: self.pages.setCurrentWidget(self.progress_view)
        )
        self.completion_button.clicked.connect(
            lambda: self.pages.setCurrentWidget(self.completion_view)
        )
        self.setup_button.clicked.connect(lambda: self.pages.setCurrentWidget(self.setup_view))
        self.settings_button.clicked.connect(
            lambda: self.pages.setCurrentWidget(self.settings_view)
        )
        self.setup_view.refresh_requested.connect(self.refresh_setup)
        self.setup_view.repair_requested.connect(self._repair_setup)
        self.setup_view.continue_requested.connect(
            lambda: self.pages.setCurrentWidget(self.project_editor)
        )
        self.project_editor.start_requested.connect(self.start_pipeline)
        self.project_editor.project_creation_requested.connect(self.create_project_structure)
        self.project_editor.dataset_review_requested.connect(
            lambda: self.pages.setCurrentWidget(self.dataset_review)
        )
        self.dataset_review.override_requested.connect(self._save_dataset_override)
        self.dataset_review.refinement_approval_requested.connect(self._submit_refinement_approval)
        self.progress_view.cancel_requested.connect(self.cancel_pipeline)
        self.progress_view.resume_requested.connect(self.resume_pipeline)
        self.completion_view.promote_requested.connect(self.promote_alternative)
        self.completion_view.copy_requested.connect(self.copy_output)
        self.completion_view.open_output_requested.connect(self.open_output)
        self.completion_view.duplicate_requested.connect(self.duplicate_project)
        self.completion_view.new_project_requested.connect(self._new_project)
        self.settings_view.save_requested.connect(self.save_destinations)

    def refresh_setup(self, *, initial: bool = False) -> None:
        try:
            checks = self.controller.setup_checks()
            self.setup_view.set_checks(checks)
        except Exception as error:
            self.setup_view.set_checks(
                (
                    {
                        "name": "Setup checks",
                        "status": "error",
                        "detail": f"{type(error).__name__}: {error}",
                        "repairable": False,
                    },
                )
            )
        if initial:
            target = self.project_editor if self.setup_view.ready else self.setup_view
            self.pages.setCurrentWidget(target)

    def _load_destinations(self) -> None:
        settings = getattr(self.controller, "settings", None)
        destinations = getattr(settings, "destinations", ())
        if isinstance(destinations, Sequence) and not isinstance(destinations, (str, bytes)):
            self.settings_view.set_destinations(destinations)

    def _new_project(self) -> None:
        if self._thread is not None:
            self.statusBar().showMessage(
                "Wait for the active pipeline to stop before starting another."
            )
            return
        self.project_editor.clear_form()
        self.pages.setCurrentWidget(self.project_editor)

    def refresh_gpus(self) -> None:
        try:
            self.project_editor.set_gpus(self.controller.discover_gpus())
        except Exception as error:
            self.project_editor.set_gpus(())
            self.project_editor.validation_message.setText(
                f"GPU discovery failed: {type(error).__name__}: {error}"
            )

    def refresh_recent_projects(self) -> None:
        try:
            self._recent_values = [as_mapping(item) for item in self.controller.recent_projects()]
        except Exception as error:
            self._recent_values = []
            self.statusBar().showMessage(
                f"Could not load recent projects: {type(error).__name__}: {error}"
            )
        self._render_recent(self._recent_values)

    def create_project_structure(self, name: str) -> None:
        """Create a name-only project tree without starting analysis or training."""

        if self._thread is not None:
            self.project_editor.project_creation_failed(
                "wait for the active pipeline to finish before adding another project"
            )
            return
        method = getattr(self.controller, "create_project", None)
        if not callable(method):
            self.project_editor.project_creation_failed(
                "Application Services did not provide project creation"
            )
            return
        self.project_editor.set_project_creation_busy(True)
        self._launch_action(
            lambda: method(name),
            self._project_creation_succeeded,
            self._project_creation_failed,
        )

    def start_pipeline(self, config: ProjectConfig) -> None:
        if self._thread is not None:
            self.project_editor.validation_message.setText(
                "A pipeline is already running. Cancel it or wait for completion."
            )
            return
        self.project_editor.set_busy(True)
        self._active_config = config
        self.dataset_review.project_id = config.project_id
        self.dataset_review.set_run_locked(True)
        self.progress_view.begin(config.project_id)
        self.pages.setCurrentWidget(self.progress_view)
        self.pipeline_started.emit(config.project_id)
        self._launch_worker(
            PipelineWorker(self.controller, "run", config=config), config.project_id
        )

    def resume_pipeline(self, project_id: str) -> None:
        if self._thread is not None or not project_id:
            return
        self._active_config = None
        self.dataset_review.project_id = project_id
        self.dataset_review.set_run_locked(True)
        self.progress_view.begin(project_id, resumed=True)
        self.pages.setCurrentWidget(self.progress_view)
        self.pipeline_started.emit(project_id)
        self._launch_worker(
            PipelineWorker(self.controller, "resume", project_id=project_id), project_id
        )

    def cancel_pipeline(self) -> None:
        if self._thread is None:
            return
        self.progress_view.mark_cancel_requested()
        try:
            self.controller.cancel_current()
        except Exception as error:
            self.progress_view.mark_failed(
                f"Cancellation failed: {type(error).__name__}: {error}", recoverable=True
            )

    def promote_alternative(self, project_id: str, checkpoint_id: str) -> None:
        self.completion_view.set_action_busy(True, "Promoting the selected checkpoint safely…")
        self._launch_action(
            lambda: self.controller.promote_alternative(project_id, checkpoint_id),
            self._promotion_succeeded,
            lambda message: self.completion_view.action_failed(
                f"Could not promote checkpoint: {message}"
            ),
        )

    def copy_output(self, project_id: str, destination_kind: str) -> None:
        self.completion_view.set_action_busy(True, f"Copying to {destination_kind} safely…")
        self._launch_action(
            lambda: self.controller.copy_output(project_id, destination_kind),
            lambda result: self._copy_succeeded(result, destination_kind),
            lambda message: self.completion_view.action_failed(f"Copy failed: {message}"),
        )

    def save_destinations(self, destinations: Sequence[Mapping[str, Any]]) -> None:
        try:
            self.controller.save_destinations(destinations)
            self.settings_view.saved()
        except Exception as error:
            self.settings_view.save_failed(
                f"Could not save destinations: {type(error).__name__}: {error}"
            )

    def open_output(self, project_id: str) -> None:
        open_method = getattr(self.controller, "open_output_folder", None)
        if not callable(open_method):
            self.completion_view.action_failed(
                "Application Services did not provide the output-folder operation."
            )
            return
        try:
            open_method(project_id)
            self.completion_view.action_succeeded("Opened the output folder.")
        except Exception as error:
            self.completion_view.action_failed(
                f"Could not open output folder: {type(error).__name__}: {error}"
            )

    def duplicate_project(self, project_id: str) -> None:
        duplicate_method = getattr(self.controller, "duplicate_project", None)
        if not callable(duplicate_method):
            recent = next(
                (
                    item
                    for item in self._recent_values
                    if str(item.get("project_id", item.get("id", ""))) == project_id
                ),
                None,
            )
            if recent is None:
                self.completion_view.action_failed(
                    "The project settings are not available for duplication."
                )
                return
            self.project_editor.load_project(recent)
        else:
            try:
                duplicate = duplicate_method(project_id)
                self.project_editor.load_project(duplicate)
            except Exception as error:
                self.completion_view.action_failed(
                    f"Could not duplicate settings: {type(error).__name__}: {error}"
                )
                return
        self.pages.setCurrentWidget(self.project_editor)

    def _launch_worker(self, worker: PipelineWorker, project_id: str) -> None:
        thread = QThread(self)
        thread.setObjectName(f"pipeline-{project_id}")
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.event_received.connect(self._on_pipeline_event)
        worker.completed.connect(self._on_pipeline_completed)
        worker.failed.connect(self._on_pipeline_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda current=thread: self._worker_thread_finished(current))
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _launch_action(
        self,
        action: Callable[[], object],
        on_success: Callable[[object], None],
        on_failure: Callable[[str], None],
    ) -> None:
        if self._action_thread is not None:
            on_failure("another background action is already running")
            return
        thread = QThread(self)
        thread.setObjectName("application-action")
        worker = ApplicationActionWorker(action)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(on_success)
        worker.failed.connect(on_failure)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda current=thread: self._action_thread_finished(current))
        thread.finished.connect(thread.deleteLater)
        self._action_thread = thread
        self._action_worker = worker
        thread.start()

    def _promotion_succeeded(self, result: object) -> None:
        self.completion_view.promotion_succeeded(as_mapping(result))

    def _project_creation_succeeded(self, result: object) -> None:
        data = as_mapping(result)
        project_id = str(data.get("project_id", ""))
        project_root = str(data.get("project_root", ""))
        self.dataset_review.project_id = project_id
        self.project_editor.project_creation_succeeded(
            project_root,
            created=bool(data.get("created", False)),
        )
        self.refresh_recent_projects()
        self.statusBar().showMessage(f"Project is ready at {project_root}")

    def _project_creation_failed(self, message: str) -> None:
        self.project_editor.set_project_creation_busy(False)
        self.project_editor.project_creation_failed(message)

    def _copy_succeeded(self, result: object, destination_kind: str) -> None:
        data = as_mapping(result)
        target = data.get("destination", data.get("path", destination_kind))
        self.completion_view.action_succeeded(f"Copied safely to {target}.")

    def _on_pipeline_event(self, raw: object) -> None:
        event = PipelineUpdate.from_value(raw)
        self.progress_view.apply_event(self._progress_event_for_view(event))
        dataset_items = event.details.get("dataset_items", event.details.get("items"))
        if isinstance(dataset_items, Sequence) and not isinstance(dataset_items, (str, bytes)):
            self.dataset_review.set_items(dataset_items)

    @staticmethod
    def _progress_event_for_view(event: PipelineUpdate) -> dict[str, object]:
        """Keep path-free Runtime Codex progress readable at the GUI boundary."""

        details = dict(event.details)
        message = event.message
        event_type = event.event_type.casefold()
        if event_type == "codex_image_progress":
            current = details.get("images_current", details.get("image_index"))
            total = details.get("images_total", details.get("image_count"))
            action = str(details.get("action", "preparing")).casefold()
            if isinstance(current, int) and isinstance(total, int) and total > 0:
                verb = "Preparing" if action == "preparing" else "Prepared"
                message = f"{verb} Codex images {current}/{total}"
        elif event_type == "codex_batch_progress":
            batch_index = details.get("batch_index")
            batch_count = details.get("batch_count")
            if isinstance(batch_index, int) and isinstance(batch_count, int) and batch_count > 0:
                message = f"Codex image batch {batch_index + 1}/{batch_count}"
        return {
            "event_type": event.event_type,
            "message": message,
            "stage": event.stage,
            "overall_progress": event.overall_progress,
            "stage_progress": event.stage_progress,
            "details": details,
        }

    def _on_pipeline_completed(self, raw: object) -> None:
        data = as_mapping(raw)
        status = str(data.get("status", "completed")).casefold()
        if status == "awaiting_review":
            project_id = str(data.get("project_id", self.dataset_review.project_id))
            self.progress_view.mark_completed()
            self.statusBar().showMessage("Training is waiting for Trigger Word or review approval.")
            self._load_refinement_review(project_id)
            return
        if status in {"cancelled", "canceled", "failed_recoverable"}:
            self.progress_view.mark_cancelled(
                str(data.get("message", "Pipeline stopped safely and can be resumed."))
            )
            return
        if self._active_config is not None:
            data.setdefault("trigger", self._active_config.trigger_token)
            data.setdefault("preset", self._active_config.preset.value)
            data.setdefault("base_model", str(self._active_config.base_model))
        self.progress_view.mark_completed()
        self.completion_view.set_result(data)
        self.pages.setCurrentWidget(self.completion_view)
        self.pipeline_completed.emit(data)

    @Slot(str, bool)
    def _on_pipeline_failed(self, message: str, recoverable: bool) -> None:
        self.progress_view.mark_failed(message, recoverable=recoverable)
        self.pages.setCurrentWidget(self.progress_view)
        self.pipeline_failed.emit(message)

    def _worker_thread_finished(self, thread: QThread) -> None:
        if self._thread is thread:
            self._thread = None
            self._worker = None
            self.project_editor.set_busy(False)
            self.dataset_review.set_run_locked(False)
            self.refresh_recent_projects()

    def _action_thread_finished(self, thread: QThread) -> None:
        if self._action_thread is thread:
            self._action_thread = None
            self._action_worker = None
            self.setup_view.set_repair_busy(False)
            self.completion_view.set_action_busy(False)
            self.project_editor.set_project_creation_busy(False)

    def _repair_setup(self, check_name: str) -> None:
        repair_method = getattr(self.controller, "repair_setup", None)
        if not callable(repair_method):
            self.statusBar().showMessage(
                f"Automatic repair is not available for {check_name}. See the check details."
            )
            return
        self.setup_view.set_repair_busy(True, check_name)
        self._launch_action(
            lambda: repair_method(check_name),
            lambda result: self._repair_succeeded(result),
            self._repair_failed,
        )

    def _repair_succeeded(self, result: object) -> None:
        del result
        try:
            self.refresh_setup()
            self.refresh_gpus()
            self.statusBar().showMessage("Setup repair completed.")
        finally:
            self.setup_view.set_repair_busy(False)

    def _repair_failed(self, message: str) -> None:
        self.setup_view.set_repair_busy(False)
        self.refresh_setup()
        self.statusBar().showMessage(f"Repair failed: {message}")

    def _save_dataset_override(self, override: Mapping[str, Any]) -> None:
        method = getattr(self.controller, "set_dataset_override", None)
        if not callable(method):
            self.statusBar().showMessage(
                "Override is retained in this review view; persistence is unavailable."
            )
            return
        try:
            method(self.dataset_review.project_id, dict(override))
        except Exception as error:
            self.statusBar().showMessage(
                f"Could not save dataset override: {type(error).__name__}: {error}"
            )

    def _load_refinement_review(self, project_id: str) -> None:
        method = getattr(self.controller, "refinement_review", None)
        if not callable(method):
            self.statusBar().showMessage("Application Services did not provide refinement review.")
            return
        self.dataset_review.project_id = project_id
        self._launch_action(
            lambda: method(project_id),
            self._refinement_review_loaded,
            self._refinement_review_load_failed,
        )

    def _refinement_review_loaded(self, raw: object) -> None:
        data = as_mapping(raw)
        data.setdefault("project_id", self.dataset_review.project_id)
        self.dataset_review.set_run_locked(False)
        self.dataset_review.set_refinement_review(data)
        self.pages.setCurrentWidget(self.dataset_review)

    def _refinement_review_load_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Could not load refinement review: {message}")

    def _submit_refinement_approval(self, approval: Mapping[str, Any]) -> None:
        method = getattr(self.controller, "submit_refinement_review", None)
        if not callable(method):
            self.dataset_review.refinement_error.setText(
                "Application Services did not provide refinement approval."
            )
            return
        project_id = self.dataset_review.project_id
        self._pending_refinement_project_id = project_id
        self._launch_action(
            lambda: method(project_id, dict(approval)),
            self._refinement_approval_saved,
            self._refinement_approval_failed,
        )

    def _refinement_approval_saved(self, _result: object) -> None:
        project_id = self._pending_refinement_project_id
        self._pending_refinement_project_id = ""
        self.statusBar().showMessage("Approval saved. Resuming the same run.")
        self.resume_pipeline(project_id)

    def _refinement_approval_failed(self, message: str) -> None:
        self._pending_refinement_project_id = ""
        self.dataset_review.refinement_error.setText(
            f"Could not save refinement approval: {message}"
        )
        self.dataset_review.approve_refinement_button.setEnabled(False)

    def _filter_recent(self, statuses: set[str]) -> None:
        filtered = [
            item
            for item in self._recent_values
            if str(item.get("status", "")).casefold() in statuses
        ]
        self._render_recent(filtered)

    def _render_recent(self, values: Sequence[Mapping[str, Any]]) -> None:
        self.recent_list.clear()
        for value in values:
            name = str(
                value.get("lora_name", value.get("name", value.get("project_id", "Project")))
            )
            status = str(value.get("status", "unknown"))
            item = QListWidgetItem(f"{name}\n{status}")
            item.setData(Qt.ItemDataRole.UserRole, dict(value))
            self.recent_list.addItem(item)

    def _activate_recent(self, item: QListWidgetItem) -> None:
        raw = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(raw, Mapping):
            return
        value = dict(raw)
        project_id = str(value.get("project_id", value.get("id", "")))
        self.dataset_review.project_id = project_id
        status = str(value.get("status", "")).casefold()
        if status in {"completed", "ready"}:
            completion = value.get("completion", value)
            if isinstance(completion, Mapping):
                self.completion_view.set_result({"project_id": project_id, **completion})
            else:
                self.completion_view.set_result(value)
            self.pages.setCurrentWidget(self.completion_view)
        elif status in {
            "active",
            "running",
            "failed",
            "failed_recoverable",
            "failed_fatal",
            "cancelled",
        }:
            self.progress_view.project_id = project_id
            self.progress_view.resume_button.setEnabled(
                status not in {"active", "running", "failed_fatal"}
            )
            self.pages.setCurrentWidget(self.progress_view)
        elif status == "awaiting_review":
            self._load_refinement_review(project_id)
        else:
            self.project_editor.load_project(value)
            self.pages.setCurrentWidget(self.project_editor)

    @staticmethod
    def _nav_button(text: str, object_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        return button

    def closeEvent(self, event: Any) -> None:
        if self._action_thread is not None:
            event.ignore()
            self.statusBar().showMessage(
                "A safe copy, promotion, or setup repair is still running. Close after it finishes."
            )
            return
        if self._thread is not None:
            self.cancel_pipeline()
            self._thread.quit()
            if not self._thread.wait(3000):
                event.ignore()
                self.statusBar().showMessage(
                    "Pipeline is still stopping safely; close again after it reaches a boundary."
                )
                return
        event.accept()
