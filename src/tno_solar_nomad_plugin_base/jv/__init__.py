from nomad.config.models.plugins import ParserEntryPoint, SchemaPackageEntryPoint


class JVParserEntryPoint(ParserEntryPoint):
    def load(self):
        from tno_solar_nomad_plugin_base.jv.jv_parser import JVParser

        return JVParser(**self.dict())


jv_parser_entry = JVParserEntryPoint(
    name='JVParser',
    description='Parser for JV measurements',
    mainfile_name_re='.*\.dat',
)


class JVPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from tno_solar_nomad_plugin_base.jv.jv_schema import m_package

        return m_package


jv_schema_entry = JVPackageEntryPoint(
    name='JVSchema',
    description='Schema for JV measurements',
)
