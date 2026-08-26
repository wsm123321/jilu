from pathlib import Path


def test_gate3_result_discloses_known_protocol_deviations():
    text=(Path(__file__).resolve().parents[1]/'THIRD_STEP_RESULT.md').read_text(encoding='utf-8')
    for phrase in ('参数化 bootstrap 未实现','Magnitude-SE Gate','风险区间未按 seed 聚类','尚非完全自适应失配轨迹'):
        assert phrase in text
