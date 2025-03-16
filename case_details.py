from fastapi import HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from database import get_db_connection
from psycopg2.extras import RealDictCursor

class CaseDetails(BaseModel):
    id: Optional[int] = None
    inquiry: str = ""
    name: str = ""
    mobile_number: str = ""
    email_address: str = ""
    appointment_date_time: Optional[datetime] = None
    category_text: str = ""
    divide_text: str = ""

def insert_case_details(case_details: CaseDetails):
    if not all([case_details.inquiry, case_details.name, case_details.mobile_number, case_details.email_address, case_details.appointment_date_time]):
        return False

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO case_details (inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        case_details.inquiry,
        case_details.name,
        case_details.mobile_number,
        case_details.email_address,
        case_details.appointment_date_time,
        case_details.category_text,
        case_details.divide_text
    ))
    conn.commit()
    cur.close()
    conn.close()
    return True

async def get_case_details():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT COUNT(*) AS count FROM case_details")
    count = cur.fetchone()  # count will be a dict like {'count': 42}
    cur.close()
    conn.close()
    return count

async def create_case_detail(case_detail: CaseDetails):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("""
        INSERT INTO case_details (inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id, inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text
        """, (
            case_detail.inquiry,
            case_detail.name,
            case_detail.mobile_number,
            case_detail.email_address,
            case_detail.appointment_date_time,
            case_detail.category_text,
            case_detail.divide_text
        ))
        new_case_detail = cur.fetchone()
        conn.commit()
        return CaseDetails(**new_case_detail)
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

async def read_case_detail(case_id: int):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM case_details WHERE id = %s", (case_id,))
        case_detail = cur.fetchone()
        if case_detail is None:
            raise HTTPException(status_code=404, detail="Case detail not found")
        return CaseDetails(**case_detail)
    finally:
        cur.close()
        conn.close()

async def update_case_detail(case_id: int, case_detail: CaseDetails):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
        UPDATE case_details
        SET inquiry = %s, name = %s, mobile_number = %s, email_address = %s, 
            appointment_date_time = %s, category_text = %s, divide_text = %s
        WHERE id = %s
        RETURNING id, inquiry, name, mobile_number, email_address, appointment_date_time, category_text, divide_text
        """, (
            case_detail.inquiry,
            case_detail.name,
            case_detail.mobile_number,
            case_detail.email_address,
            case_detail.appointment_date_time,
            case_detail.category_text,
            case_detail.divide_text,
            case_id
        ))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Case detail not found")
        columns = [desc[0] for desc in cur.description]
        updated_case_detail = dict(zip(columns, row))
        conn.commit()
        return updated_case_detail
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

async def delete_case_detail(case_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM case_details WHERE id = %s RETURNING id", (case_id,))
        deleted_case = cur.fetchone()
        if deleted_case is None:
            raise HTTPException(status_code=404, detail="Case detail not found")
        conn.commit()
        return {"message": f"Case detail with id {case_id} has been deleted"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

async def read_case_details(skip: int = Query(0, ge=0), limit: int = Query(10, ge=1, le=100)):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute("SELECT * FROM case_details ORDER BY created_at DESC OFFSET %s LIMIT %s", (skip, limit))
        case_details = cur.fetchall()
        return [CaseDetails(**case) for case in case_details]
    finally:
        cur.close()
        conn.close()