import logging
logger = logging.getLogger(__name__)

import pandas as pd
from tqdm.autonotebook import tqdm
import numpy as np
import healpy as hp
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from histpy import Histogram, HealpixAxis, Axis, Axes
from scoords import SpacecraftFrame

from cosipy.spacecraftfile import SpacecraftAxisMap

class SpacecraftAttitudeExposureTable(pd.DataFrame):
    """
    A class to analyze exposure time per each spacecraft attitude
    
    Table columns are:
    - scatt_binning_index: int 
    - healpix_index: list of tuple. Each tuple is (healpix_index_zpointing, healpix_index_xpointing).
    - zpointing: np.array of [l, b] in degrees. Array of z-pointings assigned to each scatt bin.
    - xpointing: np.array of [l, b] in degrees. Array of x-pointings assigned to each scatt bin.
    - zpointing_averaged: [l, b] in degrees. Averaged z-pointing in each scatt bin.
    - xpointing_averaged: [l, b] in degrees. Averaged x-pointing in each scatt bin.
    - earth_zenith: np.array of [l, b] in degrees. Array of earth zenith assigned to each scatt bin.
    - altitude: float in km. The satellite altitude assigned to each scatt bin.
    - delta_time: np.array of float in second. Exposure times for pointings assigned to each scatt bin.
    - exposure: float in second. total exposure for each scatt bin.
    - num_pointings: number of pointings assigned to each scatt bin.
    - bkg_group: index of the backgroud group. will be used in the data analysis.

    Attributes
    ----------
    df : :py:class:`pd.DataFrame`
        pandas dataframe with the above columns
    nside : int
        Healpix NSIDE parameter.
    scheme : str, default 'ring'
        Healpix scheme. Either 'ring', 'nested'. 
    """

    def __init__(self, df, nside, scheme = 'ring'):

        super().__init__(pd.DataFrame(df))

        self.nside = nside

        if scheme == 'ring' or scheme == 'nested':
            self.scheme = scheme
        else:
            logger.warning('The scheme should be "ring" or "nested" in SpacecraftAttitudeExposureTable. It will be set to "ring".')
            self.scheme = 'ring'

    def __eq__(self, other):
        for name in ['scatt_binning_index', 'healpix_index', 'exposure', 'num_pointings', 'bkg_group']:
            if not np.all(self[name] == other[name]):
                return False
        
        for name in ['delta_time', 'zpointing', 'xpointing', 'zpointing_averaged', 'xpointing_averaged', 'earth_zenith', 'altitude']:
            for self_, other_ in zip(self[name], other[name]):
                if not np.all(self_ == other_):
                    return False

        return (self.nside == other.nside) and (self.scheme == other.scheme)

    @classmethod
    def from_orientation(cls, orientation, nside, scheme = 'ring', start = None, stop = None, min_exposure = None, min_num_pointings = None):
        """
        Produce exposure table from orientation.

        Parameters
        ----------
        orientation : :py:class:`cosipy.spacecraftfile.SpacecraftFile` 
            Orientation
        nside : int
            Healpix NSIDE parameter.
        scheme : str, default 'ring'
            Healpix scheme. Either 'ring', 'nested'.
        start : :py:class:`astropy.time.Time` or None, default None
            Start time to analyze the orientation
        stop : :py:class:`astropy.time.Time` or None, default None
            Stop time to analyze the orientation
        min_exposure : float or None, default None
            Minimum exposure time required for each scatt bin
        min_num_pointings : int or None, default None
            Minimum number of pointings required for each scatt bin

        Returns
        -------
        :py:class:`cosipy.spacecraftfile.SpacecraftAttitudeExposureTable`
        """
        
        df = cls.analyze_orientation(orientation, nside, scheme, start, stop, min_exposure, min_num_pointings)

        new = cls(df, nside, scheme)

        return new

    # GTI should be a mandary parameter
    @classmethod
    def analyze_orientation(cls, orientation, nside, scheme = 'ring', start = None, stop = None, min_exposure = None, min_num_pointings = None):
        """
        Produce pd.DataFrame from orientation.

        Parameters
        ----------
        orientation : :py:class:`cosipy.spacecraftfile.SpacecraftFile` 
            Orientation
        nside : int
            Healpix NSIDE parameter.
        scheme : str, default 'ring'
            Healpix scheme. Either 'ring', 'nested'.
        start : :py:class:`astropy.time.Time` or None, default None
            Start time to analyze the orientation
        stop : :py:class:`astropy.time.Time` or None, default None
            Stop time to analyze the orientation
        min_exposure : float or None, default None
            Minimum exposure time required for each scatt bin
        min_num_pointings : int or None, default None
            Minimum number of pointings required for each scatt bin

        Returns
        -------
        :py:class:`pd.DataFrame`
        """

        logger.info(f'angular resolution: {hp.nside2resol(nside) * 180 / np.pi} deg.')    
    
        indices_healpix = [] # (idx_z, idx_x)
        delta_times = []
        xpointings = [] # [l_x, b_x]
        zpointings = [] # [l_z, b_z]
        earth_zenith = [] # [earth_zenith_l, earth_zenith_b]
        altitude_list = []
                
        if start is not None and stop is not None:
            orientation = orientation.source_interval(start, stop)
        elif start is not None:
            logger.error("please specify the stop time")
            return
        elif stop is not None:
            logger.error("please specify the start time")
            return
        
        ori_time = orientation.get_time()
            
        logger.info(f'duration: {(ori_time[-1] - ori_time[0]).to("day")}')
        
        attitude = orientation.get_attitude()
        
        pointing_list = attitude.transform_to("galactic").as_axes()

        n_pointing = len(pointing_list[0])
        
        l_x = 0.5 * (orientation.x_pointings.l.value[:-1] + orientation.x_pointings.l.value[1:])
        b_x = 0.5 * (orientation.x_pointings.b.value[:-1] + orientation.x_pointings.b.value[1:])

        l_z = 0.5 * (orientation.z_pointings.l.value[:-1] + orientation.z_pointings.l.value[1:])
        b_z = 0.5 * (orientation.z_pointings.b.value[:-1] + orientation.z_pointings.b.value[1:])

        earth_zenith_l = 0.5 * (orientation.earth_zenith.l.value[:-1] + orientation.earth_zenith.l.value[1:])
        earth_zenith_b = 0.5 * (orientation.earth_zenith.b.value[:-1] + orientation.earth_zenith.b.value[1:])

        if scheme == 'ring':
            nest = False
        elif scheme == 'nested':
            nest = True
        else:
            logger.warning('Warning: the scheme should be "ring" or "nested". It was set to "ring".')
            nest = False

        idx_x = hp.ang2pix(nside, l_x, b_x, nest=nest, lonlat=True)
        idx_z = hp.ang2pix(nside, l_z, b_z, nest=nest, lonlat=True)
        
        delta_time = (ori_time[1:] - ori_time[:-1]).to('s').value
        altitude = 0.5 * (orientation.get_altitude()[1:] + orientation.get_altitude()[:-1])
        
        for i in tqdm(range(n_pointing - 1)):
            
            if (idx_z[i], idx_x[i]) in indices_healpix:
                idx = indices_healpix.index((idx_z[i], idx_x[i]))
                delta_times[idx].append(delta_time[i])
                xpointings[idx].append([l_x[i], b_x[i]])
                zpointings[idx].append([l_z[i], b_z[i]])            
                earth_zenith[idx].append([earth_zenith_l[i], earth_zenith_b[i]])
                altitude_list[idx].append(altitude[i])
            else:
                indices_healpix.append((idx_z[i], idx_x[i]))
                delta_times.append([delta_time[i]])
                xpointings.append([[l_x[i], b_x[i]]])
                zpointings.append([[l_z[i], b_z[i]]])
                earth_zenith.append([[earth_zenith_l[i], earth_zenith_b[i]]])
                altitude_list.append([altitude[i]])
        
        indices_scatt_binning = [i for i in range(len(indices_healpix))] 
        
        # to numpy
        zpointings = [ np.array(_) for _ in zpointings]
        xpointings = [ np.array(_) for _ in xpointings]
        earth_zenith = [ np.array(_) for _ in earth_zenith]

        zpointings_averaged = [ cls._get_averaged_pointing(z, dt) for (z, dt) in zip(zpointings, delta_times) ]
        xpointings_averaged = [ cls._get_averaged_pointing(x, dt) for (x, dt) in zip(xpointings, delta_times) ]

        exposures = [ np.sum(np.array(_)) for _ in delta_times]
        num_pointings = [ len(_) for _ in delta_times]
        bkg_groups = [ 0 for i in delta_times]
        
        df = pd.DataFrame(data = {'scatt_binning_index': indices_scatt_binning, 'healpix_index': indices_healpix, 
                                  'zpointing': zpointings,
                                  'xpointing': xpointings,
                                  'zpointing_averaged': zpointings_averaged,
                                  'xpointing_averaged': xpointings_averaged,
                                  'earth_zenith': earth_zenith,
                                  'altitude': altitude_list, 
                                  'delta_time': delta_times, 
                                  'exposure': exposures,
                                  'num_pointings': num_pointings,
                                  'bkg_group': bkg_groups})
        
        if min_exposure is not None:
            df = df[df['exposure'] >= min_exposure]

        if min_num_pointings is not None:
            df = df[df['num_pointings'] >= min_num_pointings]
        
        if min_exposure is not None or min_num_pointings is not None:
            df['scatt_binning_index'] = [i for i in range(len(df))] 
        
        return df 

    @classmethod
    def from_fits(cls, filename):
        """
        Read exposure table from a fits file.

        Parameters
        ----------
        filename : str
            Path to file
        
        Returns
        -------
        :py:class:`cosipy.image_deconvolution.SpacecraftAttitudeExposureTable`
        """

        infile = fits.open(filename)
        hdu = infile[1]
    
        if hdu.name != "EXPOSURETABLE":
            logger.error("cannot find EXPOSURETABLE")
            return
    
        indices_scatt_binning = hdu.data['scatt_binning_index']
        indices_healpix = [ (z, x) for (z, x) in zip(hdu.data['healpix_index_z_pointing'], hdu.data['healpix_index_x_pointing']) ]

        zpointings = [ [ [l, b] for (l, b) in zip(z_l, z_b) ] for (z_l, z_b) in zip(hdu.data['zpointing_l'], hdu.data['zpointing_b']) ]
        zpointings = [ np.array(_) for _ in zpointings]

        xpointings = [ [ [l, b] for (l, b) in zip(x_l, x_b) ] for (x_l, x_b) in zip(hdu.data['xpointing_l'], hdu.data['xpointing_b']) ]
        xpointings = [ np.array(_) for _ in xpointings]

        earth_zeniths = [ [ [l, b] for (l, b) in zip(x_l, x_b) ] for (x_l, x_b) in zip(hdu.data['earth_zenith_l'], hdu.data['earth_zenith_b']) ]
        earth_zeniths = [ np.array(_) for _ in earth_zeniths]

        zpointings_averaged = [ np.array([z_ave_l, z_ave_b]) for (z_ave_l, z_ave_b)  in zip(hdu.data['zpointing_averaged_l'], hdu.data['zpointing_averaged_b']) ]

        xpointings_averaged = [ np.array([x_ave_l, x_ave_b]) for (x_ave_l, x_ave_b)  in zip(hdu.data['xpointing_averaged_l'], hdu.data['xpointing_averaged_b']) ]

        altitudes = np.array(hdu.data['altitude'])

        delta_times = np.array(hdu.data['delta_time'])

        exposures = hdu.data['exposure']

        num_pointings = hdu.data['num_pointings']

        bkg_groups = hdu.data['bkg_group']
        
        df = pd.DataFrame(data = {'scatt_binning_index': indices_scatt_binning, 'healpix_index': indices_healpix, 
                                  'zpointing': zpointings,
                                  'xpointing': xpointings,
                                  'zpointing_averaged': zpointings_averaged,
                                  'xpointing_averaged': xpointings_averaged,
                                  'earth_zenith': earth_zeniths,
                                  'altitude': altitudes, 
                                  'delta_time': delta_times, 
                                  'exposure': exposures,
                                  'num_pointings': num_pointings,
                                  'bkg_group': bkg_groups})

        nside = hdu.header['NSIDE']
        scheme = hdu.header['SCHEME']
        
        new = cls(df, nside, scheme)

        return new

    def save_as_fits(self, filename, overwrite = False):
        """
        Save exposure table as a fits file.

        Parameters
        ----------
        filename : str
            Path to file
        overwrite : bool, default False
        """

        # primary HDU
        primary_hdu = fits.PrimaryHDU()
    
        #exposure table
        names = ['scatt_binning_index', 'exposure', 'num_pointings', 'bkg_group']
        formats = ['K', 'D', 'K', 'K']
        units = ['', 's', '', '']
        
        columns = [ fits.Column(name=names[i], array=self[names[i]].to_numpy(), format = formats[i], unit = units[i]) 
                     for i in range(len(names))]
        
        column_healpix_index_z_pointing = fits.Column(name='healpix_index_z_pointing', 
                                                      array=np.array([idx[0] for idx in self['healpix_index']]), format = 'K')
        column_healpix_index_x_pointing = fits.Column(name='healpix_index_x_pointing', 
                                                      array=np.array([idx[1] for idx in self['healpix_index']]), format = 'K')
        
        columns.append(column_healpix_index_z_pointing)
        columns.append(column_healpix_index_x_pointing)

        column_altitude = fits.Column(name='altitude', format='PD()', unit = 'km',
                                        array=np.array(self['altitude'].array, dtype=np.object_))
        columns.append(column_altitude)    
    
        column_delta_time = fits.Column(name='delta_time', format='PD()', unit = 's',
                                        array=np.array(self['delta_time'].array, dtype=np.object_))
        columns.append(column_delta_time)    
        
        column_zpointing_l = fits.Column(name='zpointing_l', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[0] for pointing in pointings] for pointings in self['zpointing']], dtype=np.object_))
        columns.append(column_zpointing_l)    
    
        column_zpointing_b = fits.Column(name='zpointing_b', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[1] for pointing in pointings] for pointings in self['zpointing']], dtype=np.object_))
        columns.append(column_zpointing_b)   
    
        column_xpointing_l = fits.Column(name='xpointing_l', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[0] for pointing in pointings] for pointings in self['xpointing']], dtype=np.object_))
        columns.append(column_xpointing_l)    
    
        column_xpointing_b = fits.Column(name='xpointing_b', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[1] for pointing in pointings] for pointings in self['xpointing']], dtype=np.object_))
        columns.append(column_xpointing_b)  

        column_zpointing_averaged_l = fits.Column(name='zpointing_averaged_l', format='D', unit = 'degree',
                                                  array=np.array([_[0] for _ in self['zpointing_averaged']]))
        columns.append(column_zpointing_averaged_l)    

        column_zpointing_averaged_b = fits.Column(name='zpointing_averaged_b', format='D', unit = 'degree',
                                                  array=np.array([_[1] for _ in self['zpointing_averaged']]))
        columns.append(column_zpointing_averaged_b)    

        column_xpointing_averaged_l = fits.Column(name='xpointing_averaged_l', format='D', unit = 'degree',
                                                  array=np.array([_[0] for _ in self['xpointing_averaged']]))
        columns.append(column_xpointing_averaged_l)    

        column_xpointing_averaged_b = fits.Column(name='xpointing_averaged_b', format='D', unit = 'degree',
                                                  array=np.array([_[1] for _ in self['xpointing_averaged']]))
        columns.append(column_xpointing_averaged_b)    
    
        column_earth_zenith_l = fits.Column(name='earth_zenith_l', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[0] for pointing in pointings] for pointings in self['earth_zenith']], dtype=np.object_))
        columns.append(column_earth_zenith_l)    
    
        column_earth_zenith_b = fits.Column(name='earth_zenith_b', format='PD()', unit = 'degree',
                                        array=np.array([[pointing[1] for pointing in pointings] for pointings in self['earth_zenith']], dtype=np.object_))
        columns.append(column_earth_zenith_b)  
        
        table_hdu = fits.BinTableHDU.from_columns(columns) 
        table_hdu.name = 'exposuretable'

        table_hdu.header['nside'] = self.nside
        table_hdu.header['scheme'] = self.scheme
        
        #save file    
        hdul = fits.HDUList([primary_hdu, table_hdu])    
        hdul.writeto(filename, overwrite = overwrite)

    @classmethod
    def _get_averaged_pointing(cls, pointing, delta_time):
        """
        Calculate an averaged pointing from given lists of pointings and exposure time on each pointing

        Parameters
        ----------
        pointing : list of np.array
            List of pointings in degrees, e.g., [ np.array([l, b]), np.array([l, b]), ...]
        delta_time : list of float
            List of exposure time in seconds for each pointing, e.g, [ 1.0, 1.0, ...] 

        Returns
        -------
        :py:class:`np.array`
            Averaged pointing in degrees, as np.array([l, b])
        """

        averaged_vector = np.sum(hp.ang2vec(pointing.T[0], pointing.T[1], lonlat = True).T * delta_time, axis = (1))
        averaged_vector /= np.linalg.norm(averaged_vector)

        averaged_l = hp.vec2ang(averaged_vector, lonlat = True)[0][0]
        averaged_b = hp.vec2ang(averaged_vector, lonlat = True)[1][0]
    
        averaged_pointing = np.array([averaged_l, averaged_b])

        return averaged_pointing 

    def calc_pointing_trajectory_map(self):
        """
        Calculate a 2-dimensional map showing exposure time for each spacecraft attitude.

        Returns
        -------
        :py:class:`cosipy.spacecraft.SpacecraftAxisMap`

        Notes
        -----
        The default axes in SpacecraftAttitudeMap is x- and y-pointings,
        but here the spacecraft attitude is described with z- and x-pointings. 
        """
    
        map_pointing_zx = SpacecraftAxisMap(nside = self.nside,
                                            scheme = self.scheme,
                                            coordsys = 'galactic',
                                            labels = ('z', 'x'))

        # HEALPix pixel indices for axes 1 and 2 (stored as a Series of 2-tuples,
        # converted to a tuple of indices per axis)
        pix0, pix1 = tuple(zip(*self['healpix_index']))

        exposure = u.Quantity(self['exposure'].values, unit = u.s, copy=False)

        map_pointing_zx.fill(pix0, pix1, weight = exposure)
        
        return map_pointing_zx

    def get_binned_data_scatt(self, unbinned_event, psichi_binning = 'local', sparse = False):
        """
        Create binned data from unbinned events using spacecraft attitude binning.
    
        Events are grouped by spacecraft attitude (z- and x-pointing), energy, 
        Compton scattering angle, and scatter direction.
    
        Parameters
        ----------
        unbinned_event : :py:class:`cosipy.data_io.UnbinnedData`
            Unbinned event data.
        psichi_binning : str, default 'local'
            Coordinate system for PsiChi axis: 'local' or 'galactic'.
        sparse : bool, default False
            If True, use sparse array representation.
    
        Returns
        -------
        :py:class:`histpy.Histogram`
            Binned data with axes ["ScAtt", "Em", "Phi", "PsiChi"].
        """
        exposure_dict = {row['healpix_index']: row['scatt_binning_index'] for _, row in self.iterrows()}
            
        # from BinnedData.py
 
        # Get energy bins:
        energy_bin_edges = np.array(unbinned_event.energy_bins)
        
        # Get phi bins:
        number_phi_bins = int(180./unbinned_event.phi_pix_size)
        phi_bin_edges = np.linspace(0,180,number_phi_bins+1)
        
        # Define psichi axis and data for binning:
        if psichi_binning == 'galactic':
            psichi_axis = HealpixAxis(nside = unbinned_event.nside, scheme = unbinned_event.scheme, coordsys = 'galactic', label='PsiChi')
            coords = SkyCoord(l=unbinned_event.cosi_dataset['Chi galactic']*u.deg, b=unbinned_event.cosi_dataset['Psi galactic']*u.deg, frame = 'galactic')
        if psichi_binning == 'local':
            psichi_axis = HealpixAxis(nside = unbinned_event.nside, scheme = unbinned_event.scheme, coordsys = SpacecraftFrame(), label='PsiChi')
            coords = SkyCoord(lon=unbinned_event.cosi_dataset['Chi local']*u.rad, lat=((np.pi/2.0) - unbinned_event.cosi_dataset['Psi local'])*u.rad, frame = SpacecraftFrame())

        # Define scatt axis and data for binning
        n_scatt_bins = len(self)
        scatt_axis = Axis(np.arange(n_scatt_bins + 1), label='ScAtt')
        
        is_nest = True if self.scheme == 'nested' else False
        
        nside_scatt = self.nside
        
        zindex = hp.ang2pix(nside_scatt, unbinned_event.cosi_dataset['Zpointings (glon,glat)'].T[0] * 180 / np.pi, 
                            unbinned_event.cosi_dataset['Zpointings (glon,glat)'].T[1] * 180 / np.pi, nest=is_nest, lonlat=True)
        xindex = hp.ang2pix(nside_scatt, unbinned_event.cosi_dataset['Xpointings (glon,glat)'].T[0] * 180 / np.pi, 
                            unbinned_event.cosi_dataset['Xpointings (glon,glat)'].T[1] * 180 / np.pi, nest=is_nest, lonlat=True)    
        scatt_data = np.array( [ exposure_dict[(z, x)] + 0.5 if (z,x) in exposure_dict.keys() else -1 for z, x in zip(zindex, xindex)] ) # should this "0.5" be needed?
        
        # Initialize histogram:
        binned_data = Histogram([scatt_axis,
                                 Axis(energy_bin_edges*u.keV, label='Em'),
                                 Axis(phi_bin_edges*u.deg, label='Phi'),
                                 psichi_axis],
                                 sparse=sparse)

        # Fill histogram:
        binned_data.fill(scatt_data, unbinned_event.cosi_dataset['Energies']*u.keV, np.rad2deg(unbinned_event.cosi_dataset['Phi'])*u.deg, coords)    
        
        return binned_data
