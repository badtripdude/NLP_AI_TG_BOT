from .consts import DefaultConstructor
class User(DefaultConstructor):
    @staticmethod
    def main_menu():
        schema = [2]
        actions = [
            '📖Мои торренты',
            '➕Добавить торрент'
        ]
        return User._create_kb(actions, schema)