## FT-gpt2-xl on test (xl_r16_v1_FT-gpt2-xl_test.json)

### good

- `dataset_fact-test-00001-heldout` (dataset_fact, ROUGE-L 1.00)
  - Q: How did month 3 look for fraud?
  - answer: Month 3 had 150,936 applications and 1,392 frauds, a fraud rate of 0.92%.
  - reference: Month 3 had 150,936 applications and 1,392 frauds, a fraud rate of 0.92%.
- `dataset_fact-test-00006-heldout` (dataset_fact, ROUGE-L 1.00)
  - Q: Which housing status setting has the worst fraud rate?
  - answer: BA has the highest fraud rate for housing status: 3.52% (4,571 frauds among 129,775 applications).
  - reference: BA has the highest fraud rate for housing status: 3.52% (4,571 frauds among 129,775 applications).
- `dataset_fact-test-00008-heldout` (dataset_fact, ROUGE-L 1.00)
  - Q: Which feature is the strongest on its own?
  - answer: Housing status catches the most fraud on its own: 17.6% of frauds at a 5% false-positive rate.
  - reference: Housing status catches the most fraud on its own: 17.6% of frauds at a 5% false-positive rate.

### bad

- `policy-test-00038-heldout` (policy, ROUGE-L 0.11)
  - Q: How risky is device operating system according to these facts?
  - answer: Device operating system is risky if it is Android or iOS, and if it has been recently updated. It is not risky if it is a desktop or laptop operating system.
  - reference: The provided facts do not include device operating system, so I cannot answer that from them. They cover only the other features listed.
  - fails: not_in_context_honest
- `dataset_fact-test-00031-heldout` (dataset_fact, ROUGE-L 0.21)
  - Q: How rare is fraud in the development months overall?
  - answer: The provided facts do not include development months 0-5, so I cannot answer that from them. They cover only months 0-5.
  - reference: Across development months 0-5 there are 794,989 applications and 8,151 frauds, a fraud rate of 1.03%. That is about 96.5 legitimate applications per fraud.
  - fails: ref_numbers_all_present
- `dataset_fact-test-00041-heldout` (dataset_fact, ROUGE-L 0.22)
  - Q: Where is the line for flagging an application?
  - answer: The line for flagging an application is in the context where it is explained: "If you see this, it means we did not find any false positives and the model stays at or above the threshold."
  - reference: The review threshold is 0.7462. It was set for a 5% false-positive rate on validation month 5; applications at or above it are flagged for review.
  - fails: ref_numbers_all_present

### banking_qa / general: best and worst ROUGE-L

- `banking_qa-test-00013-heldout` (banking_qa, ROUGE-L 1.00)
  - Q: What is checked during due diligence on a customer?
  - answer: Customer due diligence (CDD) means collecting and checking information about a customer and the purpose of the account, so the bank can understand the risk the customer poses.
  - reference: Customer due diligence (CDD) means collecting and checking information about a customer and the purpose of the account, so the bank can understand the risk the customer poses.
- `general-test-00039` (general, ROUGE-L 0.00)
  - Q: Expand the following Abbreviation: "IMO"
  - answer: "International Mineral Operations" This abbreviation is used to describe the mining and refining of minerals from the earth.
  - reference: IMO stands for "In My Opinion."
