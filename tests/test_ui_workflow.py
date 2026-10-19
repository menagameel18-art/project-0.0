"""
Integration tests for UI workflow, viewports synchronization, and 3D rendering initialization.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when running or debugging directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_and_demo_case_workflow(qapp):
    window = MainWindow()
    assert window is not None

    # Load instant procedural demo case
    window.load_demo_case()

    # Verify that volume processor has valid data
    proc = window.volume_processor
    assert proc.has_data()
    assert proc.geometry.num_z == 64
    assert proc.geometry.num_x == 128
    assert proc.geometry.num_y == 128

    # Verify that all 2D viewports have rendered QImages
    vp = window.viewports
    assert vp.axial_view.viewport.qimage is not None
    assert not vp.axial_view.viewport.qimage.isNull()

    assert vp.sagittal_view.viewport.qimage is not None
    assert not vp.sagittal_view.viewport.qimage.isNull()

    assert vp.coronal_view.viewport.qimage is not None
    assert not vp.coronal_view.viewport.qimage.isNull()

    # Verify sliders have correct range
    assert vp.axial_view.slice_slider.maximum() == 63
    assert vp.sagittal_view.slice_slider.maximum() == 127
    assert vp.coronal_view.slice_slider.maximum() == 127

    # Verify 3D volume rendering actor has volume mapper and input
    assert vp.volume_3d_view.volume_actor is not None
    assert vp.volume_3d_view.volume_mapper.GetInput() is not None

    # Test slider navigation updates slices
    vp.axial_view.slice_slider.setValue(10)
    assert vp.axial_view.current_slice == 10

    # Test crosshair linking
    vp._on_axial_crosshair_clicked(50.0, 60.0)
    assert vp.curr_x == 50
    assert vp.curr_y == 60
    assert vp.sagittal_view.current_slice == 50
    assert vp.coronal_view.current_slice == 60

    # Test Window/Level preset application
    vp.apply_window_preset("Bone")
    assert vp.axial_view.viewport.wl_manager.window == 1800.0
    assert vp.axial_view.viewport.wl_manager.level == 400.0

    # Clean close
    window.close()


if __name__ == "__main__":
    app = qapp()
    test_main_window_and_demo_case_workflow(app)
    print("All UI workflow and 3D rendering tests passed successfully!")
