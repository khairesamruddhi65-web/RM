# Dataset data dictionary

| Field | Required | Meaning |
|---|---|---|
| record_id | yes | Stable unique row identifier |
| abstract | yes | Text available at the time of application/decision; use proposal text if legally available |
| outcome | for outcome model | 1 = funded, 0 = not funded/unsuccessful under a documented definition |
| scheme_id | strongly recommended | Funding scheme identifier |
| call_id | strongly recommended | Exact call / RFP version |
| decision_year | strongly recommended | Year of decision, not project publication year |
| institution_id | optional | Pseudonymized institution/group key for grouped splits |
| project_family_id | optional | Links resubmissions and related proposals |
| source | yes for research | Source dataset/API/file and retrieval date |
| license | yes for research | License or access agreement |
| synthetic | yes in demo | True for synthetic records; exclude from scientific claims |

Do not label records as rejected just because they do not appear in a funded-awards database.
