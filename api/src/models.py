from sqlalchemy import Column, String, Integer, ForeignKey, Enum, Boolean, DateTime, ForeignKeyConstraint, ARRAY, JSON, TEXT

from sqlalchemy.orm import relationship

# class Images(Base):
#     __tablename__ = 'images'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     name = Column(String(255), nullable=False)
