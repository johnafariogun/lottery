# main.py
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import bcrypt
import asyncio
import httpx
import random
import uuid
from celery import Celery
import os



# Configuration
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI

class Settings(BaseSettings):
    secret_key: str
    database_url: str
    celery_broker: str

    model_config = SettingsConfigDict(env_file=".env") # Load from .env file

settings = Settings()




celery_broker = settings.celery_broker
database_url = settings.database_url
secret_key = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

print(secret_key, celery_broker, database_url)
# Celery setup
celery_app = Celery('lottery', broker=celery_broker)

# Database setup
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    credits = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    spins = relationship("LotterySpin", back_populates="user")

class LotterySpin(Base):
    __tablename__ = "lottery_spins"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    spin_id = Column(String, unique=True, index=True)
    cost = Column(Integer)
    prize_type = Column(String)
    prize_value = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="spins")

class BillingTransaction(Base):
    __tablename__ = "billing_transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    transaction_id = Column(String, unique=True, index=True)
    amount = Column(Float)
    status = Column(String, default="pending")
    telecom_response = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserProfile(BaseModel):
    id: int
    email: str
    username: str
    credits: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class SpinRequest(BaseModel):
    spin_type: str = "basic"  # basic, premium, mega

class SpinResult(BaseModel):
    spin_id: str
    cost: int
    prize_type: str
    prize_value: int
    remaining_credits: int

class SpinHistory(BaseModel):
    spin_id: str
    cost: int
    prize_type: str
    prize_value: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class LeaderboardEntry(BaseModel):
    username: str
    total_winnings: int
    biggest_win: int

class BillingRequest(BaseModel):
    amount: float
    phone_number: str

class Token(BaseModel):
    access_token: str
    token_type: str

# FastAPI app
app = FastAPI(title="Lottery System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication utilities
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Lottery logic
SPIN_COSTS = {"basic": 10, "premium": 50, "mega": 100}
PRIZE_POOLS = {
    "basic": [
        {"type": "credits", "value": 0, "weight": 50},
        {"type": "credits", "value": 5, "weight": 30},
        {"type": "credits", "value": 20, "weight": 15},
        {"type": "credits", "value": 100, "weight": 5},
    ],
    "premium": [
        {"type": "credits", "value": 0, "weight": 40},
        {"type": "credits", "value": 25, "weight": 35},
        {"type": "credits", "value": 100, "weight": 20},
        {"type": "credits", "value": 500, "weight": 5},
    ],
    "mega": [
        {"type": "credits", "value": 0, "weight": 30},
        {"type": "credits", "value": 50, "weight": 40},
        {"type": "credits", "value": 200, "weight": 25},
        {"type": "credits", "value": 1000, "weight": 5},
    ]
}

def calculate_prize(spin_type: str):
    prizes = PRIZE_POOLS[spin_type]
    weights = [p["weight"] for p in prizes]
    selected_prize = random.choices(prizes, weights=weights)[0]
    return selected_prize["type"], selected_prize["value"]

# Celery tasks
@celery_app.task
def process_telecom_billing(transaction_id: str, phone_number: str, amount: float):
    """Background task to process telecom billing"""
    try:
        # Simulate async call to telecom API
        response = asyncio.run(call_telecom_api(phone_number, amount))
        
        # Update transaction status in database
        db = SessionLocal()
        transaction = db.query(BillingTransaction).filter(
            BillingTransaction.transaction_id == transaction_id
        ).first()
        if transaction:
            transaction.status = "completed" if response["success"] else "failed"
            transaction.telecom_response = str(response)
            print(str(response))
            db.commit()
        db.close()
        
        return response
    except Exception as e:
        # Update transaction as failed
        db = SessionLocal()
        transaction = db.query(BillingTransaction).filter(
            BillingTransaction.transaction_id == transaction_id
        ).first()
        if transaction:
            transaction.status = "failed"
            transaction.telecom_response = str(e)
            db.commit()
        db.close()
        raise

async def call_telecom_api(phone_number: str, amount: float):
    """Mock telecom API call"""
    await asyncio.sleep(2)  # Simulate network delay
    
    # Mock response - 90% success rate
    success = random.random() > 0.1
    
    return {
        "success": success,
        "transaction_id": str(uuid.uuid4()),
        "phone_number": phone_number,
        "amount": amount,
        "message": "Payment processed successfully" if success else "Payment failed"
    }

# API Endpoints

@app.get("/")
def read_root():
    return {"message": "Lottery System API", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Authentication endpoints
@app.post("/auth/register", response_model=UserProfile)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(User).filter(
        (User.email == user.email) | (User.username == user.username)
    ).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    
    # Create new user
    hashed_password = hash_password(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        credits=100  # Welcome bonus
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/auth/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# User endpoints
@app.get("/user/profile", response_model=UserProfile)
def get_profile(current_user: User = Depends(get_current_user)):
    return current_user

# Lottery endpoints
@app.post("/lottery/spin", response_model=SpinResult)
def spin_lottery(spin_request: SpinRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    spin_type = spin_request.spin_type
    
    if spin_type not in SPIN_COSTS:
        raise HTTPException(status_code=400, detail="Invalid spin type")
    
    cost = SPIN_COSTS[spin_type]
    
    if current_user.credits < cost:
        raise HTTPException(status_code=400, detail="Insufficient credits")
    
    # Deduct credits
    current_user.credits -= cost
    
    # Calculate prize
    prize_type, prize_value = calculate_prize(spin_type)
    
    # Award prize
    if prize_type == "credits":
        current_user.credits += prize_value
    
    # Create spin record
    spin_id = str(uuid.uuid4())
    spin = LotterySpin(
        user_id=current_user.id,
        spin_id=spin_id,
        cost=cost,
        prize_type=prize_type,
        prize_value=prize_value
    )
    
    db.add(spin)
    db.commit()
    
    return SpinResult(
        spin_id=spin_id,
        cost=cost,
        prize_type=prize_type,
        prize_value=prize_value,
        remaining_credits=current_user.credits
    )

@app.get("/lottery/history", response_model=List[SpinHistory])
def get_spin_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    spins = db.query(LotterySpin).filter(
        LotterySpin.user_id == current_user.id
    ).order_by(desc(LotterySpin.created_at)).limit(50).all()
    
    return spins

@app.get("/lottery/leaderboard", response_model=List[LeaderboardEntry])
def get_leaderboard(db: Session = Depends(get_db)):
    # Get top winners by total winnings
    from sqlalchemy import func
    
    leaderboard = db.query(
        User.username,
        func.sum(LotterySpin.prize_value).label('total_winnings'),
        func.max(LotterySpin.prize_value).label('biggest_win')
    ).join(LotterySpin).group_by(User.id, User.username).order_by(
        func.sum(LotterySpin.prize_value).desc()
    ).limit(10).all()
    
    return [
        LeaderboardEntry(
            username=entry.username,
            total_winnings=entry.total_winnings or 0,
            biggest_win=entry.biggest_win or 0
        )
        for entry in leaderboard
    ]

# Billing endpoints
@app.post("/billing/pay")
def process_payment(
    billing_request: BillingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Create billing transaction
    transaction_id = str(uuid.uuid4())
    transaction = BillingTransaction(
        user_id=current_user.id,
        transaction_id=transaction_id,
        amount=billing_request.amount,
        status="pending"
    )
    db.add(transaction)
    db.commit()
    
    # Queue background task
    process_telecom_billing.delay(
        transaction_id,
        billing_request.phone_number,
        billing_request.amount
    )
    
    return {
        "message": "Payment initiated",
        "transaction_id": transaction_id,
        "status": "pending"
    }

# Admin endpoints
@app.get("/admin/reports")
def get_daily_reports(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Simple admin check - in production, use proper role-based access
    # if current_user.username != "admin":
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    today = datetime.utcnow().date()
    
    # Daily statistics
    from sqlalchemy import func
    
    daily_spins = db.query(func.count(LotterySpin.id)).filter(
        func.date(LotterySpin.created_at) == today
    ).scalar()
    
    daily_revenue = db.query(func.sum(LotterySpin.cost)).filter(
        func.date(LotterySpin.created_at) == today
    ).scalar() or 0
    
    daily_payouts = db.query(func.sum(LotterySpin.prize_value)).filter(
        func.date(LotterySpin.created_at) == today
    ).scalar() or 0
    
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    
    return {
        "date": today,
        "daily_spins": daily_spins,
        "daily_revenue": daily_revenue,
        "daily_payouts": daily_payouts,
        "net_profit": daily_revenue - daily_payouts,
        "active_users": active_users
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)