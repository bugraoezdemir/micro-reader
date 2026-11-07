"""Quantity.to takes a quantity's own unit by the same spellings as the target (review
cycle 2, R2-13: Quantity(60000, 'ms').to('s') failed as 'a length and a time')."""
import pytest

from micro_reader import Quantity


@pytest.mark.parametrize("value, unit, to, want, want_unit", [
    (60000, "ms", "s", 60.0, "second"), (1, "nm", "um", 0.001, "micrometer"),
    (1, "s", "min", 1 / 60, "minute"), (1, "micron", "nm", 1000.0, "nanometer"),
    (2, "micrometer", "um", 2.0, "micrometer")])
def test_own_unit_spellings_convert(value, unit, to, want, want_unit):
    got = Quantity(value, unit).to(to)
    assert got.unit == want_unit and got.value == pytest.approx(want)


def test_the_reason_is_the_real_one():
    with pytest.raises(ValueError, match="a time and a length"):
        Quantity(1, "ms").to("um")
    with pytest.raises(ValueError, match="'furlong' is no unit micro-reader knows"):
        Quantity(1, "furlong").to("um")
