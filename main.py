# main.py
from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
# from google.oauth2 import id_token
# from google.auth.transport import requests

import jwt
import bcrypt
import asyncio
import httpx
import random
import uuid
from celery import Celery
import os
import secrets


# Configuration

from fastapi import FastAPI
from settings import settings
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid_configuration"





celery_broker = settings.celery_broker
database_url = settings.database_url
secret_key = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Celery setup
celery_app = Celery('lottery', broker=celery_broker)

# Database setup
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class GoogleAuthRequest(BaseModel):
    code: str
    state: Optional[str] = None

class GoogleUserInfo(BaseModel):
    id: str
    email: EmailStr
    verified_email: bool
    name: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: str
    locale: Optional[str] = None

class GoogleTokenResponse(BaseModel):
    access_token: str
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str
    token_type: str
    id_token: str

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)  # Nullable for OAuth users
    credits = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Google OAuth fields
    google_id = Column(String, unique=True, nullable=True, index=True)
    provider = Column(String, default="local")  # local, google, facebook, etc.
    profile_picture = Column(String, nullable=True)
    is_email_verified = Column(Boolean, default=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    
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
# Add session middleware to main app
from starlette.middleware.sessions import SessionMiddleware

# Add to your main.py
app.add_middleware(SessionMiddleware, secret_key=secret_key)
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

class GoogleAuthService:
    """Google OAuth authentication service."""
    
    def __init__(self):
        self.client_id = settings.google_client_id 
        self.client_secret = settings.google_client_secret
        self.redirect_uri = settings.google_redirect_uri
        
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Google OAuth credentials not configured")
    
    async def get_google_oauth_config(self) -> Dict[str, str]:
        """Get Google OAuth configuration from discovery document."""
        async with httpx.AsyncClient() as client:
            response = await client.get(GOOGLE_DISCOVERY_URL)
            response.raise_for_status()
            return response.json()
    
    def generate_oauth_url(self, state: Optional[str] = None) -> str:
        """Generate Google OAuth authorization URL."""
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"https://accounts.google.com/o/oauth2/auth?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access tokens."""
        token_url = "https://oauth2.googleapis.com/token"
        
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to exchange code for tokens: {response.text}"
                )
            
            # return GoogleTokenResponse(**response.json())
            return response.json()

    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Google API."""
        userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(userinfo_url, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user information from Google"
                )
            
            # return GoogleUserInfo(**response.json())
            return response.json()

    
    def verify_id_token(self, id_token_str: str) -> Dict[str, Any]:
        """Verify Google ID token (simplified - in production use google-auth library)."""
        try:
            # payload = id_token.verify_oauth2_token(
            #     id_token_str, requests.Request(), self.client_id
            # )
            payload = jwt.decode(id_token_str, options={"verify_signature": False, "verify_exp": False })
            return payload
            # payload = jwt.decode(id_token, options={"verify_signature": False})
            
            # # Verify audience
            # if payload.get("aud") != self.client_id:
            #     raise HTTPException(
            #         status_code=status.HTTP_400_BAD_REQUEST,
            #         detail="Invalid token audience"
            #     )
            
            # # Verify expiration
            # if payload.get("exp", 0) < datetime.utcnow().timestamp():
            #     raise HTTPException(
            #         status_code=status.HTTP_400_BAD_REQUEST,
            #         detail="Token has expired"
            #     )
            
            # return payload
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ID token: {str(e)}"
            )

# Update main.py with Google OAuth endpoints
from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse

# Initialize Google Auth Service
google_auth = GoogleAuthService()

# Create OAuth router
oauth_router = APIRouter(prefix="/auth", tags=["authentication"])

@oauth_router.get("/google/login")
async def google_login(request: Request):
    """Initiate Google OAuth login."""
    # Generate state parameter for CSRF protection
    state = secrets.token_urlsafe(32)
    
    # Store state in session (you might want to use Redis for this)
    request.session["oauth_state"] = state
    
    oauth_url = google_auth.generate_oauth_url(state=state)
    return {"authorization_url": oauth_url}

@oauth_router.get("/google/callback")
async def google_callback(
    request: Request,
    # auth_request: GoogleAuthRequest = Depends(),
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db)
):
    """Handle Google OAuth callback."""
    # Verify state parameter
    stored_state = request.session.get("oauth_state")
    if not stored_state or stored_state != state: #auth_request.state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid state parameter"
        )
    
    try:
        # Exchange code for tokens
        tokens = await google_auth.exchange_code_for_tokens(code)
        
        # Get user info
        user_info = await google_auth.get_user_info(tokens["access_token"])
        
        # Verify ID token
        id_token_payload = google_auth.verify_id_token(tokens["id_token"])
        
        # Find or create user
        user = db.query(User).filter(
            (User.google_id == user_info["id"]) | (User.email == user_info["email"])
        ).first()
        
        if user:
            # Update existing user
            user.google_id = user_info["id"]
            user.provider = "google"
            user.profile_picture = user_info["picture"]
            user.is_email_verified = user_info["verified_email"]
            user.first_name = user_info["given_name"]
            user.last_name = user_info["family_name"]
        else:
            # Create new user
            username = user_info["email"].split("@")[0]
            # Ensure username is unique
            counter = 1
            original_username = username
            while db.query(User).filter(User.username == username).first():
                username = f"{original_username}{counter}"
                counter += 1
            
            user = User(
                email=user_info["email"],
                username=username,
                google_id=user_info["id"],
                provider="google",
                is_email_verified=user_info["verified_email"],
                first_name=user_info.get("given_name"),
                last_name=user_info.get("family_name"),
                profile_picture=user_info.get("picture"),
                credits=100  # Welcome bonus
            )
            db.add(user)
        
        db.commit()
        db.refresh(user)
        
        # Create JWT token
        access_token = create_access_token(data={"sub": user.username})
        
        # Clear session state
        request.session.pop("oauth_state", None)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "profile_picture": user.profile_picture,
                "credits": user.credits,
                "provider": user.provider
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth authentication failed: {str(e)}"
        )

@oauth_router.post("/google/token")
async def google_token_auth(
    token_request: dict,
    db: Session = Depends(get_db)
):
    """Authenticate with Google ID token directly (for mobile apps)."""
    id_token = token_request.get("id_token")
    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID token is required"
        )
    
    try:
        # Verify ID token
        payload = google_auth.verify_id_token(id_token)
        
        # Extract user info from token
        google_id = payload.get("sub")
        email = payload.get("email")
        name = payload.get("name", "")
        picture = payload.get("picture")
        given_name = payload.get("given_name")
        family_name = payload.get("family_name")
        email_verified = payload.get("email_verified", False)
        
        # Find or create user
        user = db.query(User).filter(
            (User.google_id == google_id) | (User.email == email)
        ).first()
        
        if user:
            # Update existing user
            user.google_id = google_id
            user.provider = "google"
            user.profile_picture = picture
            user.is_email_verified = email_verified
            user.first_name = given_name
            user.last_name = family_name
        else:
            # Create new user
            username = email.split("@")[0]
            counter = 1
            original_username = username
            while db.query(User).filter(User.username == username).first():
                username = f"{original_username}{counter}"
                counter += 1
            
            user = User(
                email=email,
                username=username,
                google_id=google_id,
                provider="google",
                profile_picture=picture,
                is_email_verified=email_verified,
                first_name=given_name,
                last_name=family_name,
                credits=100  # Welcome bonus
            )
            db.add(user)
        
        db.commit()
        db.refresh(user)
        
        # Create JWT token
        access_token = create_access_token(data={"sub": user.username})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "profile_picture": user.profile_picture,
                "credits": user.credits,
                "provider": user.provider
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Token authentication failed: {str(e)}"
        )

@oauth_router.post("/logout")
async def logout(request: Request, current_user: User = Depends(get_current_user)):
    """Logout user and clear session."""
    # Clear session
    request.session.clear()
    
    return {"message": "Successfully logged out"}


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

app.include_router(oauth_router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)