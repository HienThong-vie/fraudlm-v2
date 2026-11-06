## FT-gpt2-xl on demo67 (xl_r16_v1_FT-gpt2-xl_demo67.json)

### good

- `demo67-00002-app931225` (explain_score, ROUGE-L 1.00)
  - Q: Which factor matters most for this application?
  - answer: The strongest factor is housing status (BA), which raises the model score compared with a typical application. The other main factors are application velocity over 4 weeks (3145) and keep-alive session (no).
  - reference: The strongest factor is housing status (BA), which raises the model score compared with a typical application. The other main factors are application velocity over 4 weeks (3145) and keep-alive session (no).
- `demo67-00003-app830382` (explain_score, ROUGE-L 1.00)
  - Q: Which factors does the model point to?
  - answer: The main factors are: device operating system (windows), which raises the model score; distinct emails on the device over 8 weeks (2), which raises the model score; and application velocity over 4 weeks (3400), which lowers the model score. Raises and lowers are measured compared with a typical application.
  - reference: The main factors are: device operating system (windows), which raises the model score; distinct emails on the device over 8 weeks (2), which raises the model score; and application velocity over 4 weeks (3400), which lowers the model score. Raises and lowers are measured compared with a typical appl
- `demo67-00005-app822349` (explain_score, ROUGE-L 1.00)
  - Q: Briefly, what does the model say about this application?
  - answer: Result: flagged for review (score 0.7833, threshold 0.7462); the leading factor is housing status (BA).
  - reference: Result: flagged for review (score 0.7833, threshold 0.7462); the leading factor is housing status (BA).

### bad

- `demo67-00147-app949756` (explain_score, ROUGE-L 0.55)
  - Q: Can you explain the role of housing status?
  - answer: Housing status is BA for this application and it raises the model score compared with a typical application. It is one of the main factors, but no single factor proves anything about the application.
  - reference: Here housing status is BA, and that raises the model score of 0.7729 compared with a typical application. Read it together with the other main factors rather than on its own.
  - fails: explain_faithful; score_correct; ref_numbers_all_present
- `demo67-00072-app929116` (explain_score, ROUGE-L 0.57)
  - Q: How much should I read into name-email similarity?
  - answer: Name-email similarity is 0.878 for this application and it lowers the model score compared with a typical application. It is one of the main factors, but no single factor proves anything about the application.
  - reference: Here name-email similarity is 0.878, and that lowers the model score of 0.0306 compared with a typical application. Read it together with the other main factors rather than on its own.
  - fails: explain_faithful; score_correct; ref_numbers_all_present
- `demo67-00088-app998231` (explain_score, ROUGE-L 0.57)
  - Q: Does name-email similarity push the score up or down?
  - answer: Name-email similarity is 0.0448 for this application and it raises the model score compared with a typical application. It is one of the main factors, but no single factor proves anything about the application.
  - reference: Here name-email similarity is 0.0448, and that raises the model score of 0.7702 compared with a typical application. Read it together with the other main factors rather than on its own.
  - fails: explain_faithful; score_correct; ref_numbers_all_present

### banking_qa / general: best and worst ROUGE-L

(none)
