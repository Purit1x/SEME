from flask import Blueprint

bp = Blueprint('mri', __name__)

from . import routes 