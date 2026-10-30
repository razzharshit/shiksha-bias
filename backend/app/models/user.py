from sqlalchemy import Column, String, Boolean, Enum as SQLEnum
from app.models.base import BaseModel
from app.models.enums import UserRole


class User(BaseModel):
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False, default="Curriculum Editor")
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.CURRICULUM_EDITOR)
    is_active = Column(Boolean, default=True)

