import os
import random as rn
from faker import Faker
from faker.providers import company, address, bank

from gen.enums import DateFormats


class CounterParty(object):

    def __init__(self, name, language, address, date_format, template=None):
        # NUMBER IDENTIFIER INSTEAD OF NAME IN RAW CTPY
        # CTPY NAME ON US?

        self.language: str = language
        self.name: str = name
        self.address: str = address
        self.date_format: str = date_format

        self.template: str = template

    def __str__(self):
        return self.name


class CounterPartyGenerator(object):
    @staticmethod
    def generate(number_of_cptys) -> list:
        cptys = []
        confirmation_templates = _get_confirmation_templates()
        date_formats = [date_format.value for date_format in DateFormats]

        fake = Faker()
        fake.add_provider(company)
        fake.add_provider(address)

        for _ in range(number_of_cptys):
            cpty_name = fake.company() + ' bank'
            random_address = fake.address()
            confirmation_template = rn.choice(confirmation_templates)
            date_format = rn.choice(date_formats)
            cpty = CounterParty(name=cpty_name,
                                language='eng',
                                address=random_address,
                                date_format=date_format,
                                template=confirmation_template)
            cptys.append(cpty)

        return cptys


def _get_confirmation_templates():
    os.chdir('confirmation_templates')
    confirmation_templates = []
    confirmation_files = os.listdir()
    print(confirmation_files)

    for confirmation_file in confirmation_files:
        with open(confirmation_file) as f:
            lines = f.readlines()
            confirmation = "".join(lines)
            confirmation_templates.append(confirmation)

    os.chdir('..')
    return confirmation_templates