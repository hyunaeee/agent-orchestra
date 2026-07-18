# agent-orchestra 🎻

**LangGraph 멀티에이전트 오케스트레이션 + 에이전트 메모리 + 평가 파이프라인** —
supervisor가 과제를 분해하고, 역할 에이전트들이 A2A 메시지로 협업하며,
critic이 승인할 때까지 재작성 루프를 도는 최소-완결 레퍼런스 구현입니다.

```
supervisor ──▶ researcher ──▶ writer ──▶ critic ──▶ END
                                ▲           │
                                └─ REVISE ──┘   (conditional edge, max rounds)
```

## 핵심 구성

| 구성 요소 | 구현 |
|---|---|
| **오케스트레이션** | LangGraph `StateGraph` — supervisor 계획 수립 → 순차 실행 → critic의 조건부 엣지로 재작성 루프 |
| **공유 메모리** | `SharedMemory` — 에이전트 간 A2A 메시지 로그 + 롤링 컨텍스트 윈도우 + 스크래치패드 |
| **절차적 메모리** | `ProceduralMemory` — critic이 승인 시 "교훈"을 JSON으로 영속화, 이후 실행에서 writer 프롬프트에 자동 주입 |
| **A2A 메시지** | `Envelope(sender, recipient, intent, content, ts)` — 모든 에이전트 통신이 봉투 단위로 기록·재생 가능 |
| **평가 파이프라인** | `eval/run_eval.py` — 루브릭 커버리지 judge(오프라인) + LLM-judge(API 키 시), pass rate·재작성 횟수·지연·토큰 리포트 |
| **LLM 백엔드** | `ANTHROPIC_API_KEY` 있으면 Claude, 없으면 결정적 오프라인 mock — 키 없이도 그래프·메모리·평가 전체가 동작 |

## 실행

```bash
pip install -r requirements.txt

python demo.py           # 과제 하나를 팀에 던지고 모든 hop을 출력
python eval/run_eval.py  # 평가셋 6개 태스크 일괄 실행 + 리포트
```

`demo.py` 출력 (오프라인 mock, 2번째 실행 — 절차 메모리가 주입된 상태):

```
=== FINAL DRAFT (writer) ===
## 결과 보고
### 도입 배경
...
> 이전 실행에서 배운 것 반영: 요구 항목을 섹션 헤더로 명시하면 누락이 줄어든다.

=== A2A MESSAGE LOG ===
  supervisor → *        [plan]   1) researcher가 과제의 핵심 항목을 조사한다 ...
  researcher → writer   [notes]  - 도입 배경: 과제 요구사항 ...
      writer → critic   [draft]  ## 결과 보고 ...
      critic → writer   [review] APPROVE 루브릭 전 항목 충족 ...
      critic → *        [lesson] 요구 항목을 섹션 헤더로 명시하면 누락이 줄어든다.
```

`eval/run_eval.py` 결과 (오프라인 judge):

```
[PASS] proposal-helpdesk  rounds=1 ...
[PASS] release-note       rounds=1 ...
[PASS] incident-report    rounds=1 ...
[PASS] onboarding-doc     rounds=1 ...
[PASS] ab-test-plan       rounds=1 ...
[PASS] vendor-compare     rounds=1 ...
--------------------------------------------------------
pass rate: 6/6 (100%) · avg rounds: 1.0
```

## 설계 노트

- **왜 mock 백엔드인가** — 오케스트레이션·메모리·평가는 모델 품질과 독립적으로 검증되어야
  합니다. mock은 결정적이라 CI에서 그래프 배선과 메모리 주입을 회귀 테스트할 수 있고,
  키를 넣는 순간 같은 코드가 Claude로 돌아갑니다.
- **절차적 메모리의 최소 형태** — 벡터 DB 없이 키워드 중첩 검색으로 시작했습니다.
  교훈 수가 늘면 `ProceduralMemory.relevant()`만 임베딩 검색으로 교체하면 됩니다.
- **평가가 곧 문서** — evalset의 각 태스크는 "포함: A, B, C" 형태의 루브릭을 갖고,
  judge는 최종 결과물의 루브릭 커버리지를 기계적으로 판정합니다. LLM-judge는 그 위에
  실용성 점수를 얹는 선택 레이어입니다.

## 관련 프로젝트

- [med-rag](https://github.com/hyunaeee/med-rag) — 고려대 안암병원에 납품한 온프렘 진료 보조 RAG 에이전트
- med-rag-vertex — 같은 시스템의 Vertex AI + ADK(SequentialAgent) 클라우드 포트 + LLM-judge 평가

---

© 2026 [aengdo](https://hyunaeee.github.io/aengdo-portfolio/) · MIT
