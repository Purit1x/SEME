"""
Factory classes for generating test data.

这个模块包含用于生成测试数据的工厂类定义。工厂类可以快速生成符合业务规则的测试数据，
支持批量创建和自定义属性。

主要功能:
- 生成符合数据模型约束的测试数据
- 支持批量创建多条测试数据
- 提供可定制的生成策略
- 自动处理数据库会话

使用示例:
    # 创建单个患者
    patient = PatientFactory()
    
    # 创建带自定义属性的患者
    patient = PatientFactory(
        patient_name='张三',
        age=35
    )
    
    # 批量创建多个患者
    patients = PatientFactory.create_batch(size=10)
    
    # 创建但不保存到数据库
    patient = PatientFactory.build()

性能说明:
- 使用 Faker 库生成逼真的中文测试数据
- 使用 factory.LazyFunction 实现延迟计算
- 使用 factory.Sequence 生成唯一标识
"""
import factory.alchemy
import random
from app.models import Patient, db
from faker import Faker

# 初始化Faker实例，使用中文本地化
faker = Faker(['zh_CN'])

class PatientFactory(factory.alchemy.SQLAlchemyModelFactory):
    """患者数据工厂类
    
    该工厂类用于生成Patient模型的测试数据。它继承自SQLAlchemyModelFactory，
    支持与SQLAlchemy的集成，自动处理数据库会话。
    
    属性:
        Meta: 工厂配置类
            - model: 指定关联的数据模型(Patient)
            - sqlalchemy_session: 指定使用的数据库会话
            - sqlalchemy_session_persistence: 指定会话持久化策略
            
        patient_name (str): 使用faker生成的随机中文姓名
        sex (str): 随机生成的性别，'男'或'女'
        age (int): 1-100之间的随机年龄
        id_number (str): 基于序列生成的18位身份证号
        photo_path (str|None): 照片路径，默认为None
        
    生成策略:
        - 姓名使用Faker生成逼真的中文姓名
        - 性别随机选择男/女
        - 年龄在合理范围内随机生成
        - 身份证号使用序列确保唯一性
        - 照片路径可选，默认为None
        
    注意事项:
        - 创建实例时会自动flush但不commit，需要手动commit
        - 可以通过传参覆盖默认的生成策略
        - build()方法创建不持久化的实例
        - create()方法创建并持久化实例
    """
    class Meta:
        model = Patient
        sqlalchemy_session = db.session
        # session在创建后flush但不commit，避免测试数据意外提交
        sqlalchemy_session_persistence = 'flush'

    # 使用faker生成逼真的中文姓名
    patient_name = factory.LazyFunction(lambda: faker.name())
    
    # 随机选择性别
    sex = factory.LazyFunction(lambda: random.choice(['男', '女']))
    
    # 生成1-100之间的随机年龄
    age = factory.LazyFunction(lambda: random.randint(1, 100))
    
    # 使用sequence生成唯一的18位身份证号
    id_number = factory.Sequence(lambda n: f'110101199{n:04d}0123')
    
    # 照片路径，可选字段
    photo_path = None
