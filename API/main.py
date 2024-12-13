from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class Student(BaseModel):
    id: str 
    name: str 
    gender: str
    age: int

db = [
    Student(
        id="20240001",
        name="A",
        gender="Male",
        age=20 
    )
]

app = FastAPI()

@app.post('/student_db')
def get_student(student: Student):
    db.append(student)

@app.get('/student_db/{student_id}')
def get_student(student_id: str):
    for s in db:
        if s.id == student_id:
            return s
        
    raise HTTPException(status_code=404, detail='Student not found')