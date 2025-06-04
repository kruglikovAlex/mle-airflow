# dags/alt_churn.py

import pendulum
import pandas as pd
import numpy as np

from steps.message import send_telegram_failure_message, send_telegram_success_message
from steps.churn import create_table, extract, transform, load
#from message import send_telegram_success_message, send_telegram_failure_message

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import XCom

from sqlalchemy import Table, MetaData, Column, Integer, Float, String, DateTime, UniqueConstraint, inspect

with DAG(
    dag_id='alt_churn',
    schedule='@once',
    start_date=pendulum.datetime(2025, 6, 3, tz="UTC"),
    catchup=False,
    tags=["ETL"],
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message
) as dag:
    
    create_table = PythonOperator(
        task_id="create_table",
        python_callable=create_table,
        provide_context=True
    )

    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=extract,
        provide_context=True
    )

    transform_data = PythonOperator(
        task_id="transform_data",
        python_callable=transform,
        provide_context=True
    )

    load_data = PythonOperator(
        task_id="load_data",
        python_callable=load,
        provide_context=True
    )

    create_table >> extract_data >> transform_data >> load_data