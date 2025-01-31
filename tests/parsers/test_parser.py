import logging

from nomad.datamodel import EntryArchive

from tno_solar_nomad_plugin_base.parsers.parser import NewParser


def test_parse_file():
    parser = NewParser()
    archive = EntryArchive()
    parser.parse('tests/data/example.out', archive, logging.getLogger())

    assert archive.workflow2.name == 'test'
