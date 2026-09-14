from types import SimpleNamespace

from app.common.response import success_response


def test_success_response_wraps_data_with_trace_id():
    request = SimpleNamespace(state=SimpleNamespace(trace_id="trace-123"))

    response = success_response(request, {"ok": True})

    assert response.traceId == "trace-123"
    assert response.status == "200"
    assert response.result == "Succeeded"
    assert response.error is None
    assert response.data == {"ok": True}
