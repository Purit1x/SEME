"""
工厂类定义，用于创建测试数据
"""
import factory.alchemy
import random
from app.models import Patient, db
from faker import Faker

faker = Faker(['zh_CN'])

class PatientFactory(factory.alchemy.SQLAlchemyModelFactory):
    """患者数据工厂类"""
    class Meta:
        model = Patient
        sqlalchemy_session = db.session
        sqlalchemy_session_persistence = 'flush'  # session在创建后flush但不commit

    patient_name = factory.LazyFunction(lambda: faker.name())
    sex = factory.LazyFunction(lambda: random.choice(['男', '女']))
    age = factory.LazyFunction(lambda: random.randint(1, 100))
    id_number = factory.Sequence(lambda n: f'110101199{n:04d}0123')  # 生成唯一的18位身份证号
    photo_path = None  # 可选字段，默认为None
