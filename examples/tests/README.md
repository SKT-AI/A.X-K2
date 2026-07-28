# 테스트

| 파일 | 설명 |
|------|------|
| `test_chat_template.py` | 채팅 템플릿이 OpenAI 호환 클라이언트가 보내는 메시지 형태들을 올바르게 렌더링하는지 검증합니다 |

```bash
pip install jinja2 huggingface_hub
python test_chat_template.py                            # Hub의 skt/A.X-K2
python test_chat_template.py <로컬 체크포인트 경로>
```

템플릿을 수정한 뒤에는 이 테스트를 통과하는지 확인하세요. 실패 시 비정상 종료 코드를 반환하므로 CI에 그대로 쓸 수 있습니다.
