"""OME-XML time steps: the planes' own times (Plane@DeltaT) win over the declared
TimeIncrement, as in every format (Bugra 2026-11-04), with a warning when they differ by
more than 5%. Review cycle 2, R2-04: a Micro-Manager MMStack OME-TIFF declared 1 ms ("as
fast as possible") for frames 1.12 s apart, and files with stamps but no TimeIncrement had
no time step."""
import warnings
import xml.etree.ElementTree as ET

import pytest

from micro_reader.backends._tiff_metadata import ome_image_metadata

NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _image(n_t=3, increment=None, stamps=(0.0, 1.1, 2.2), unit="s"):
    inc = f' TimeIncrement="{increment}" TimeIncrementUnit="ms"' if increment else ""
    planes = "".join(f'<Plane TheT="{t}" TheZ="{z}" TheC="0" DeltaT="{stamps[t] + 0.01 * z}" '
                     f'DeltaTUnit="{unit}"/>' for t in range(n_t) for z in range(2))
    xml = (f'<OME xmlns="{NS}"><Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" '
           f'Type="uint8" SizeX="4" SizeY="4" SizeZ="2" SizeC="1" SizeT="{n_t}"{inc}>'
           f'<Channel ID="Channel:0:0"/>{planes}</Pixels></Image></OME>')
    return ET.fromstring(xml)[0]


def _t(element, n_t=3):
    return ome_image_metadata(element, {"t": n_t, "c": 1, "z": 2, "y": 4, "x": 4}, 1, {},
                              dict).pixel_sizes.get("t")


def test_the_stamps_win_over_a_declared_increment_with_a_warning():
    with pytest.warns(UserWarning, match=r"1100 millisecond by the planes' DeltaT, 1 millisecond declared"):
        t = _t(_image(increment=1, stamps=(0.0, 1100.0, 2200.0), unit="ms"))
    assert t.unit == "millisecond" and t.value == pytest.approx(1100.0)


def test_stamps_alone_give_the_step():
    t = _t(_image())
    assert t.unit == "second" and t.value == pytest.approx(1.1)


def test_a_declared_increment_the_stamps_agree_with_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = _t(_image(increment=1080, stamps=(0.0, 1100.0, 2200.0), unit="ms"))
    assert t.value == pytest.approx(1100.0)


def test_no_stamps_keep_the_declared_increment():
    element = _image(increment=500)
    for plane in element.iter(f"{{{NS}}}Plane"):
        del plane.attrib["DeltaT"]
    assert _t(element).value == 500


def test_one_time_point_has_no_step_from_stamps():
    assert _t(_image(n_t=1, stamps=(0.0,)), n_t=1) is None
