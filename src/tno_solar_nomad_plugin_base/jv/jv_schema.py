import os

import numpy as np
import pandas as pd
from nomad.datamodel.data import EntryData
from nomad.datamodel.datamodel import EntryArchive
from nomad.datamodel.metainfo.annotations import BrowserAdaptors, ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import Measurement
from nomad.datamodel.metainfo.plot import PlotSection
from nomad.metainfo import SchemaPackage
from nomad.metainfo.metainfo import Quantity, Section, SubSection
from nomad.units import ureg as units

m_package = SchemaPackage()


class SolarCellJV(PlotSection):
    m_def = Section(
        label_quantity='cell_name',
        a_plotly_graph_object=[
            {'data': {'x': 'voltage', 'y': '#current_density'}},
            {'data': {'x': 'voltage', 'y': '#current_density'}},
        ],
    )

    data_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor=BrowserAdaptors.RawFileAdaptor),
    )

    open_circuit_voltage = Quantity(
        type=np.dtype(np.float64),
        unit='V',
        shape=[],
        description='Open circuit voltage.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    short_circuit_current_density = Quantity(
        type=np.dtype(np.float64),
        unit='mA / cm**2',
        shape=[],
        description='Short circuit current density.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    fill_factor = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='Fill factor.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    efficiency = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='Power conversion efficiency.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    max_power_point = Quantity(
        type=np.dtype(np.float64),
        unit='W/cm^2',
        shape=[],
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    series_resistance = Quantity(
        type=np.dtype(np.float64),
        unit='ohm*cm**2',
        shape=[],
        description='The series resistance as extracted from the *J-V* curve.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    shunt_resistance = Quantity(
        type=np.dtype(np.float64),
        unit='ohm*cm**2',
        shape=[],
        description='The shunt resistance as extracted from the *J-V* curve.',
        a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity),
    )

    def derive_n_values(self):
        if self.current_density is not None:
            return len(self.current_density)
        if self.voltage is not None:
            return len(self.voltage)
        else:
            return 0

    n_values = Quantity(type=int, derived=derive_n_values)

    def normalize(self, archive: 'EntryArchive', logger):
        super().normalize(archive, logger)


class SolarCellJvCurve(SolarCellJV):
    cell_name = Quantity(
        type=str,
        shape=[],
        a_eln=ELNAnnotation(component=ELNComponentEnum.StringEditQuantity),
    )

    voltage = Quantity(
        type=np.dtype(np.float64),
        shape=['n_values'],
        unit='V',
    )

    current_density = Quantity(
        type=np.dtype(np.float64),
        shape=['n_values'],
        unit='mA/cm^2',
    )

    def normalize(self, archive: 'EntryArchive', logger):
        super().normalize(archive, logger)


class BaseMeasurement(Measurement):
    m_def = Section()

    def normalize(self, archive: 'EntryArchive', logger):
        if self.samples:
            for s in self.samples:
                s.normalize(archive, logger)
        super().normalize(archive, logger)


class JVMeasurement(BaseMeasurement):
    m_def = Section(
        label_quantity='data_file',
        validate=False,
    )

    data_file = Quantity(
        type=str,
        a_eln=ELNAnnotation(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor=BrowserAdaptors.RawFileAdaptor),
    )

    active_area = Quantity(type=np.dtype(np.float64), unit=('cm^2'), a_eln=ELNAnnotation(component=ELNComponentEnum.NumberEditQuantity, defaultDisplayUnit='cm^2', props=dict(minValue=0)))

    jv_curve = SubSection(
        section_def=SolarCellJvCurve,
        repeats=True,
        label_quantity='cell_name',
    )

    def normalize(self, archive: 'EntryArchive', logger):
        self.method = 'JV Measurement'
        super().normalize(archive, logger)


class SunSim_JVMeasurement(JVMeasurement, EntryData):
    pass


class Wacom_JVMeasurement(JVMeasurement, EntryData):
    m_def = Section(
        a_eln=dict(hide=['lab_id', 'steps', 'location'], properties=dict(order=['name', 'data_file', 'active_area', 'samples'])),
        a_plot=[
            {
                'x': 'jv_curve/:/voltage',
                'y': 'jv_curve/:/current_density',
                'layout': {
                    'showlegend': True,
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
    )

    def normalize(self, archive: 'EntryArchive', logger):
        super(JVMeasurement, self).normalize(archive, logger)
        super().normalize(archive, logger)


m_package.__init_metainfo__()


def get_encoding(f):
    import chardet

    return chardet.detect(f.read())['encoding']


def convert_into_processed_dataframe(file) -> pd.DataFrame:
    lines = file.readlines()

    new_lines = lines[:-6]
    tab_separated_lines = [line.strip().replace(' ', '\t') for line in new_lines]
    df = pd.DataFrame([line.split('\t') for line in tab_separated_lines], columns=['V1', 'I1', 'V2', 'I2', 'V3', 'I3', 'V4', 'I4'])

    # Converts the values of each column to numeric (fixes deprecation error)
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass

    # This is the pixel jsc correction part! TODO 1 for now.
    #   df[['I1']] *= correction_factors['c1']
    #   df[['I2']] *= correction_factors['c2']
    #   df[['I3']] *= correction_factors['c3']
    #   df[['I4']] *= correction_factors['c4']

    # This is how I splitt the forward and reverse, V change of less than 0.001 and a current less than 1000mA (when you hit compliance the V does not change)
    index_to_split = df.index[(df['V1'].diff().abs() < 0.001) & (df['I1'] < 1000)].tolist()[0]
    dfRV = df.iloc[:index_to_split].add_suffix('_R')
    dfFW = df.iloc[index_to_split:].add_suffix('_F')
    df_combined = pd.concat([dfRV.reset_index(drop=True), dfFW.reset_index(drop=True)], axis=1)

    return df_combined


def get_jv_data(df: pd.DataFrame, pixel_index: int) -> dict:
    jv_dict = {}

    # Now processing each IV set for this file
    rsh_fits = []  # List to store Rsh fits
    rs_fits = []  # List to store Rs fits

    jv_dict['jv_curve'] = []

    selected_option_deadp = 2  # TODO

    max_power, Rsh, Rs, Jsc, Voc, FF, n = calculate_iv_parameters(df, f'V{pixel_index}_R', f'I{pixel_index}_R', selected_option_deadp)

    jv_dict['reverse'] = {'mpp': max_power, 'rsh': Rsh, 'rs': Rs, 'jsc': Jsc, 'voc': Voc, 'ff': FF, 'n': n}
    jv_dict['reverse']['jv_curve'] = {'name': f'Cell {pixel_index} Reverse', 'voltage': df[f'V{pixel_index}_R'], 'current_density': df[f'I{pixel_index}_R']}

    max_power, Rsh, Rs, Jsc, Voc, FF, n = calculate_iv_parameters(df, f'V{pixel_index}_F', f'I{pixel_index}_F', selected_option_deadp)

    jv_dict['forward'] = {'mpp': max_power, 'rsh': Rsh, 'rs': Rs, 'jsc': Jsc, 'voc': Voc, 'ff': FF, 'n': n}
    jv_dict['forward']['jv_curve'] = {'name': f'Cell {pixel_index} Forward', 'voltage': df[f'V{pixel_index}_F'], 'current_density': df[f'I{pixel_index}_F']}

    # Calculate linear fits for Rsh and Rs for Reverse IV set
    dfrshR = df.loc[
        (df[f'V{pixel_index}_R'] >= -0.10) & (df[f'V{pixel_index}_R'] < 0.25),
        [
            # Define dataframe for Reverse IV set (shunt resistance)
            f'V{pixel_index}_R',
            f'I{pixel_index}_R',
        ],
    ]
    dfrsR = df.loc[
        (df[f'I{pixel_index}_R'] <= -0.1) & (df[f'I{pixel_index}_R'] > -30),
        [
            # Define dataframe for Reverse IV set (series resistance)
            f'V{pixel_index}_R',
            f'I{pixel_index}_R',
        ],
    ]

    if not dfrshR.empty:
        XrshR = np.array(dfrshR[f'V{pixel_index}_R'])
        YrshR = np.array(dfrshR[f'I{pixel_index}_R'])
        modelrshR = np.polyfit(XrshR, YrshR, 1)
        rsh_fits.append(modelrshR)
    else:
        rsh_fits.append(None)

    if not dfrsR.empty:
        XrsR = np.array(dfrsR[f'V{pixel_index}_R'])
        YrsR = np.array(dfrsR[f'I{pixel_index}_R'])
        modelrsR = np.polyfit(XrsR, YrsR, 1)
        rs_fits.append(modelrsR)
    else:
        rs_fits.append(None)

    # Calculate linear fits for Rsh and Rs for Forward IV set
    dfrshF = df.loc[
        (df[f'V{pixel_index}_F'] >= -0.10) & (df[f'V{pixel_index}_F'] < 0.25),
        [
            # Define dataframe for Forward IV set (shunt resistance)
            f'V{pixel_index}_F',
            f'I{pixel_index}_F',
        ],
    ]
    dfrsF = df.loc[
        (df[f'I{pixel_index}_F'] <= -0.10) & (df[f'I{pixel_index}_F'] > -30),
        [
            # Define dataframe for Forward IV set (series resistance)
            f'V{pixel_index}_F',
            f'I{pixel_index}_F',
        ],
    ]

    if not dfrshF.empty:
        XrshF = np.array(dfrshF[f'V{pixel_index}_F'])
        YrshF = np.array(dfrshF[f'I{pixel_index}_F'])
        modelrshF = np.polyfit(XrshF, YrshF, 1)
        rsh_fits.append(modelrshF)
    else:
        rsh_fits.append(None)

    if not dfrsF.empty:
        XrsF = np.array(dfrsF[f'V{pixel_index}_F'])
        YrsF = np.array(dfrsF[f'I{pixel_index}_F'])
        modelrsF = np.polyfit(XrsF, YrsF, 1)
        rs_fits.append(modelrsF)
    else:
        rs_fits.append(None)

    return jv_dict


def calculate_iv_parameters(df, v_col, i_col, option):
    """
    option(int) can only take 2 values: 1 == Include dead pixels
                                   2 == Remove dead pixels
    """
    X = np.array(df[v_col])
    Y = np.array(df[i_col])
    Power = X * Y
    max_power = np.max(Power)

    # Initialize variables to handle cases where the dataset may be empty
    slope_rsh, intercept_rsh = (np.nan, np.nan)
    slope_rs, intercept_rs = (np.nan, np.nan)
    Rsh, Rs = (np.nan, np.nan)
    Jsc, Voc, FF, n = (np.nan, np.nan, np.nan, np.nan)

    # Shunt resistance section of data
    df_rsh = df[(df[v_col] >= -0.10) & (df[v_col] < 0.25)]
    # Series resistance section of data
    df_rs = df[(df[i_col] <= -0.1) & (df[i_col] > -30)]

    # Perform linear fits if data is sufficient
    if not df_rsh.empty:
        try:
            slope_rsh, intercept_rsh = np.polyfit(df_rsh[v_col], df_rsh[i_col], 1)
            # Use abs() to ensure Rsh is never negative
            Rsh = 1 / (abs(slope_rsh)) * 1000
            Jsc = np.polyval([slope_rsh, intercept_rsh], 0)
        except Exception as e:
            print(f'Error calculating shunt resistance: {e}')

    if not df_rs.empty:
        try:
            slope_rs, intercept_rs = np.polyfit(df_rs[v_col], df_rs[i_col], 1)
            # Assuming Rs should always be positive; add abs() if needed
            Rs = 1 / (slope_rs) * -1000 if slope_rs <= 0 else np.nan
            Voc = np.roots([slope_rs, intercept_rs])[0] if len(np.roots([slope_rs, intercept_rs])) > 0 else np.nan
        except Exception as e:
            print(f'Error calculating series resistance: {e}')

    match option:
        case 1:  # Include dead pixels in the statistic analysis
            # Filters out dead pixels by comparing Voc and Jsc
            if Voc < 0.15 or Voc > 2.14 or np.isnan(Voc) or Jsc > 37 or Jsc <= 1 or np.isnan(Jsc):
                Voc = 0
                Jsc = 0
                FF = 0
                n = 0
            else:  # If the pixel is not dead, compute the following:
                VJ = Voc * Jsc
                FF = max_power / VJ
                # Ensures there are no crazy negative values of FF or Voc
                if FF <= 0.3 or np.isnan(FF) or FF >= 0.97:
                    FF = 0
                    n = 0
                else:
                    n = Voc * Jsc * FF if VJ != 0 else 0
        case 2:  # Remove dead pixels from the statistic analysis
            # Filters out dead pixels by comparing Voc and Jsc
            if Voc < 0.15 or Voc > 2.14 or np.isnan(Voc) or Jsc > 37 or Jsc <= 1 or np.isnan(Jsc):
                Voc = np.nan
                Jsc = np.nan
                FF = np.nan
                n = np.nan
            else:  # If the pixel is not dead, compute the following:
                VJ = Voc * Jsc
                FF = max_power / VJ
                # Ensures there are no crazy negative values of FF or Voc
                if FF <= 0.3 or np.isnan(FF) or FF >= 0.97:
                    FF = np.nan
                    n = np.nan
                else:
                    n = Voc * Jsc * FF if VJ != 0 else np.nan
    return max_power, Rsh, Rs, Jsc, Voc, FF, n
