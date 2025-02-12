import os

from nomad.app.v1.models.models import MetadataResponse
from nomad.datamodel.data import EntryData
from nomad.datamodel.datamodel import EntryArchive
from nomad.datamodel.metainfo.basesections import CompositeSystemReference
from nomad.parsing.parser import MatchingParser

from tno_solar_nomad_plugin_base.jv.jv_schema import SolarCellJvCurve, Wacom_JVMeasurement, convert_into_processed_dataframe, get_encoding, get_jv_data


class JVParser(MatchingParser):
    def parse(self, mainfile: str, archive: 'EntryArchive', logger) -> None:
        data_file = os.path.basename(mainfile)

        with archive.m_context.raw_file(data_file, 'br') as f:
            encoding = get_encoding(f)

        with archive.m_context.raw_file(data_file, 'rt', encoding=encoding) as f:
            df = convert_into_processed_dataframe(f)

            for i in range(1, 5):
                jv_dict = get_jv_data(df, i)

                entry = Wacom_JVMeasurement()

                # get_jv_archive(jv_dict, self.data_file, self)
                entry.file_name = os.path.basename(data_file)
                entry.active_area = 0.089
                entry.jv_curve = []

                entry.jv_curve.append(
                    SolarCellJvCurve(
                        cell_name=f'P{i} rv',
                        voltage=jv_dict['reverse']['jv_curve']['voltage'],
                        current_density=jv_dict['reverse']['jv_curve']['current_density'],
                        open_circuit_voltage=jv_dict['reverse']['voc'],
                        short_circuit_current_density=jv_dict['reverse']['jsc'],
                        fill_factor=jv_dict['reverse']['ff'],
                        efficiency=jv_dict['reverse']['n'],
                        max_power_point=jv_dict['reverse']['mpp'],
                        series_resistance=jv_dict['reverse']['rs'],
                        shunt_resistance=jv_dict['reverse']['rsh'],
                    )
                )

                entry.jv_curve.append(
                    SolarCellJvCurve(
                        cell_name=f'P{i} fw',
                        voltage=jv_dict['forward']['jv_curve']['voltage'],
                        current_density=jv_dict['forward']['jv_curve']['current_density'],
                        open_circuit_voltage=jv_dict['forward']['voc'],
                        short_circuit_current_density=jv_dict['forward']['jsc'],
                        fill_factor=jv_dict['forward']['ff'],
                        efficiency=jv_dict['forward']['n'],
                        max_power_point=jv_dict['forward']['mpp'],
                        series_resistance=jv_dict['forward']['rs'],
                        shunt_resistance=jv_dict['forward']['rsh'],
                    )
                )

                # entry.data_file = data_file
                file_name = f'{os.path.basename(mainfile)}.pixel{i}.archive.json'
                create_archive(entry, archive, file_name)


def create_archive(entity: EntryData, archive: EntryArchive, file_name: str, overwrite: bool = False) -> bool:
    import json

    if not archive.m_context.raw_path_exists(file_name) or overwrite:
        entity_entry = entity.m_to_dict(with_root_def=True)
        with archive.m_context.raw_file(file_name, 'w') as f:
            json.dump({'data': entity_entry}, f)
        archive.m_context.process_updated_raw_file(file_name, allow_modify=overwrite)
        return True
    return False


def set_sample_reference(
    archive: EntryArchive,
    entry: EntryData,
    search_id: str,
    logger,
    upload_id: str = None,
):
    if upload_id is None:
        search_result = search_entry_by_id(archive, entry, search_id)
    else:
        search_result = search_sampleid_in_upload(archive, search_id, upload_id)

    if len(search_result.data) == 1:
        data = search_result.data[0]
        upload_id, entry_id = data['upload_id'], data['entry_id']

        # this is a specific workflow from HZB.
        if 'sample' in data['entry_type'].lower() or 'library' in data['entry_type'].lower():
            entry.samples = [CompositeSystemReference(reference=get_reference(upload_id, entry_id))]
    else:
        logger.warning(f'No sample found for sample id {search_id} in upload {upload_id}')


def get_reference(upload_id: str, entry_id: str) -> str:
    return f'../uploads/{upload_id}/archive/{entry_id}#data'


def search_entry_by_id(archive: 'EntryArchive', entry: 'EntryData', search_id: str) -> MetadataResponse:
    from nomad.search import search

    # TODO define our own s_id identifier for our samples
    query = {'results.eln.lab_ids': search_id}
    search_result = search(owner='all', query=query, user_id=archive.metadata.main_author.user_id)
    return search_result


def search_sampleid_in_upload(archive: 'EntryArchive', sample_id, upload_id: str) -> MetadataResponse:
    from nomad.search import search

    # TODO define our own s_id identifier for our samples
    query = {'results.eln.lab_ids': sample_id, 'upload_id': upload_id}
    search_result = search(owner='all', query=query, user_id=archive.metadata.main_author.user_id)
    return search_result
