import os

basedir = os.path.abspath(os.path.dirname(__file__))

SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'users.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = 'votre_clef_secrete_super_secure'
