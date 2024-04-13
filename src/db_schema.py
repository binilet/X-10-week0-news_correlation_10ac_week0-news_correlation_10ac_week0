from sqlalchemy import Column, Integer, String, Text, Float, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Article(Base):
    __tablename__ = 'articles'

    article_id = Column(String(255), primary_key=True)
    source_id = Column(String(255))
    source_name = Column(String(255))
    author = Column(String(255))
    title = Column(Text)
    description = Column(Text)
    url = Column(Text)
    url_to_image = Column(Text)
    published_at = Column(TIMESTAMP(timezone=True))
    content = Column(Text)
    category = Column(String(255))
    article = Column(Text)
    title_sentiment = Column(Float)

class TrafficData(Base):
    __tablename__ = 'traffic_data'

    traffic_id = Column(Integer, primary_key=True)
    GlobalRank = Column(Integer)
    TldRank = Column(Integer)
    Domain = Column(String(255), unique=True)
    TLD = Column(String(10))
    RefSubNets = Column(Integer)
    RefIPs = Column(Integer)
    IDN_Domain = Column(String(255))
    IDN_TLD = Column(String(10))
    PrevGlobalRank = Column(Integer)
    PrevTldRank = Column(Integer)
    PrevRefSubNets = Column(Integer)
    PrevRefIPs = Column(Integer)

class DomainLocation(Base):
    __tablename__ = 'domain_location'

    location_id = Column(Integer, primary_key=True)
    SourceCommonName = Column(String(255))
    location = Column(String(2))
    traffic_data_domain = Column(String(255), ForeignKey('traffic_data.Domain'))

    traffic_data = relationship("TrafficData", back_populates="domain_location")
