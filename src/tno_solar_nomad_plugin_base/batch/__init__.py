from nomad.config.models.plugins import ParserEntryPoint, SchemaPackageEntryPoint


class BatchParserEntryPoint(ParserEntryPoint):
    def load(self):
        from tno_solar_nomad_plugin_base.batch.batch_parser import BatchParser

        return BatchParser(**self.dict())


batch_parser_entry = BatchParserEntryPoint(
    name='BatchParser',
    description='Parser for batches',
    mainfile_name_re='^(.+\.xlsx)$',
)


class BatchPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from tno_solar_nomad_plugin_base.batch.batch_schema import m_package

        return m_package


batch_schema_entry = BatchPackageEntryPoint(
    name='BatchSchema',
    description='Schema for experimental batches.',
)
