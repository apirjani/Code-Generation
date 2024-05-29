# Encoder-Decoder Models vs. LLMs for Code Generation

## Project Description

The goal of this project is to compare several models' performances on the task of translating English prompts querying a flight database to SQL queries that will be run on a real flight database. The 3 models implemented and evaluated are a pre-trained T5 finetuned on a dataset of English to SQL query translations, a T5 trained from scratch, and Gemma-7B with k-shot prompting.

Since different SQL queries can correctly correspond to the same natural language instruction, we perform evaluation by executing the SQL queries directly on a database of flight information, which is described by `flight_database.schema`. We record 3 separate metrics: Record F1, Record Exact Match and SQL Query Exact Match (EM). The F1 score is computed between database records produced by model-generated and ground-truth SQL queries. Record EM, on the other hand, is a stricter metric checking whether the produced records match exactly, similar to an accuracy score. Finally, SQL Query EM will tell us whether the SQL queries themselves match exactly. We ultimately use Record F1 to rank the performances of the different models.

## Test Sample

Natural Language Query: cheapest fare from indianapolis to seattle

Generated SQL: `SELECT DISTINCT fare_1.fare_id FROM fare fare_1 , flight_fare flight_fare_1 , flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE fare_1.one_direction_cost =( SELECT MIN( fare_1.one_direction_cost ) FROM fare fare_1 , flight_fare flight_fare_1 , flight flight_1 , airport_service airport_service_1 , city city_1 , airport_service airport_service_2 , city city_2 WHERE fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'INDIANAPOLIS' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'SEATTLE' ) AND fare_1.fare_id = flight_fare_1.fare_id AND flight_fare_1.flight_id = flight_1.flight_id AND flight_1.from_airport = airport_service_1.airport_code AND airport_service_1.city_code = city_1.city_code AND city_1.city_name = 'INDIANAPOLIS' AND flight_1.to_airport = airport_service_2.airport_code AND airport_service_2.city_code = city_2.city_code AND city_2.city_name = 'SEATTLE'`

## Results (Reported as Record F1 computed over test set)

- T5 finetuned: 0.64741
- T5 trained from scratch: 0.45245
- Gemma-7B (8-shot, refer to `prompting.py` to see the prompt used): 0.44486

## Execution Instructions

To execute t5_ft, simply run 'python train_t5 --finetune' and set any other tags and hyperparameters you'd like. Please refer to the `train_t5.py` file to see the valid arguments.

To execute t5_scr, follow the instructions above and leave out the "--finetune" tag.

To execute the prompting code, simply run 'python prompting.py' with any tags and hyperparameters you'd like. Please refer to the `prompting.py` file to see the valid arguments.

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:

```
python evaluate.py
  --predicted_sql results/t5_ft_dev.sql
  --predicted_records records/t5_ft_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```
