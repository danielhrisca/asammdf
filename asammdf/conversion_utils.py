# -*- coding: utf-8 -*-
'''
asammdf utility functions for channel conversions
'''

from . import v2_v3_constants as v3c
from . import v2_v3_blocks as v3b
from . import v4_constants as v4c
from . import v4_blocks as v4b

__all__ = [
    'conversion_transfer',
]


def conversion_transfer(conversion, version=3):
    """ convert between mdf4 and mdf3 channel conversions

    Parameters
    ----------
    conversion : block
        channel conversion
    version : int
        target mdf version

    Returns
    -------
    conversion : block
        channel conversion for specified version

    """

    if version <= 3:
        if conversion is None:
            conversion = v3b.ChannelConversion(
                conversion_type=v3c.CONVERSION_TYPE_NONE,
            )
        else:
            conversion_type = conversion['conversion_type']
            if conversion['id'] == b'CC':
                pass
            else:
                unit = conversion.unit.strip(' \r\n\t\0').encode('latin-1')

                if conversion_type == v4c.CONVERSION_TYPE_NON:
                    conversion = dict(conversion)
                    conversion['conversion_type'] = v3c.CONVERSION_TYPE_NONE
                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        **conversion
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_LIN:
                    conversion = dict(conversion)
                    conversion['conversion_type'] = v3c.CONVERSION_TYPE_LINEAR
                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        **conversion
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_RAT:
                    conversion = dict(conversion)
                    conversion['conversion_type'] = v3c.CONVERSION_TYPE_RAT
                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        **conversion
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_TAB:
                    conversion = dict(conversion)
                    conversion['conversion_type'] = v3c.CONVERSION_TYPE_TAB
                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        **conversion
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_TABI:
                    conversion = dict(conversion)
                    conversion['conversion_type'] = v3c.CONVERSION_TYPE_TABI
                    conversion['ref_param_nr'] = conversion['val_param_nr'] // 2
                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        **conversion
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_ALG:
                    formula = conversion.formula.replace('X', 'X1')
                    conversion = dict(conversion)
                    conversion['conversion_type'] = v3c.CONVERSION_TYPE_FORMULA
                    conversion = v3b.ChannelConversion(
                        formula=formula,
                        unit=unit,
                        **conversion
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_RTAB:
                    nr = (conversion['val_param_nr'] - 1) // 3
                    kargs = {
                        'ref_param_nr': 2 * nr,
                        'conversion_type': v3c.CONVERSION_TYPE_TABI,
                    }

                    for i in range(nr):
                        l_ = conversion['lower_{}'.format(i)]
                        u_ = conversion['upper_{}'.format(i)]
                        p_ = conversion['phys_{}'.format(i)]
                        kargs['raw_{}'.format(2 * i)] = l_
                        kargs['raw_{}'.format(2 * i + 1)] = u_ - 0.000001
                        kargs['phys_{}'.format(2 * i)] = p_
                        kargs['phys_{}'.format(2 * i + 1)] = p_

                    conversion = v3b.ChannelConversion(
                        unit=unit,
                        **kargs
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_TABX:
                    nr = conversion['val_param_nr']
                    kargs = {
                        'ref_param_nr': nr,
                        'unit': unit,
                        'conversion_type': v3c.CONVERSION_TYPE_TABX,
                    }
                    for i in range(nr):
                        kargs['param_val_{}'.format(i)] = conversion['val_{}'.format(i)]
                        kargs['text_{}'.format(i)] = conversion.referenced_blocks['text_{}'.format(i)]['text']

                    conversion = v3b.ChannelConversion(
                        **kargs
                    )

                elif conversion_type == v4c.CONVERSION_TYPE_RTABX:
                    nr = conversion['val_param_nr'] // 2
                    kargs = {
                        'ref_param_nr': nr + 1,
                        'unit': unit,
                        'conversion_type': v3c.CONVERSION_TYPE_RTABX,
                    }
                    for i in range(nr):
                        kargs['lower_{}'.format(i)] = conversion['lower_{}'.format(i)]
                        kargs['upper_{}'.format(i)] = conversion['upper_{}'.format(i)]
                        kargs['text_{}'.format(i)] = conversion.referenced_blocks['text_{}'.format(i)]['text']

                    new_conversion = v3b.ChannelConversion(
                        **kargs
                    )

                    new_conversion.referenced_blocks['default_addr'] = v3b.TextBlock(
                        text=conversion.referenced_blocks['default_addr']['text'],
                    )

                    conversion = new_conversion

    else:
        if conversion is None or conversion['id'] == b'##CC':
            pass
        else:
            conversion_type = conversion['conversion_type']
            unit = conversion['unit'].decode('latin-1').strip(' \r\n\t\0')
            if conversion_type == v3c.CONVERSION_TYPE_NONE:
                conversion = dict(conversion)
                conversion['conversion_type'] = v4c.CONVERSION_TYPE_NON
                conversion = v4b.ChannelConversion(
                    **conversion
                )

            elif conversion_type == v3c.CONVERSION_TYPE_LINEAR:
                conversion = dict(conversion)
                conversion['conversion_type'] = v4c.CONVERSION_TYPE_LIN
                conversion = v4b.ChannelConversion(
                    **conversion
                )

            elif conversion_type == v3c.CONVERSION_TYPE_RAT:
                conversion = dict(conversion)
                conversion['conversion_type'] = v4c.CONVERSION_TYPE_RAT
                conversion = v4b.ChannelConversion(
                    **conversion
                )

            elif conversion_type == v3c.CONVERSION_TYPE_FORMULA:
                formula = conversion['formula'].decode('latin-1')
                conversion = dict(conversion)
                conversion['conversion_type'] = v4c.CONVERSION_TYPE_ALG
                conversion = v4b.ChannelConversion(
                    **conversion
                )
                conversion.formula = formula

            elif conversion_type == v3c.CONVERSION_TYPE_TAB:
                conversion = dict(conversion)
                conversion['conversion_type'] = v4c.CONVERSION_TYPE_TAB
                conversion = v4b.ChannelConversion(
                    **conversion
                )

            elif conversion_type == v3c.CONVERSION_TYPE_TABI:
                conversion = dict(conversion)
                conversion['conversion_type'] = v4c.CONVERSION_TYPE_TABI
                conversion['val_param_nr'] = conversion['ref_param_nr'] * 2
                conversion['ref_param_nr'] = 0
                conversion = v4b.ChannelConversion(
                    **conversion
                )

            elif conversion_type == v3c.CONVERSION_TYPE_TABX:

                nr = conversion['ref_param_nr']
                kargs = {
                    'val_param_nr': nr,
                    'ref_param_nr': nr + 1,
                    'conversion_type': v4c.CONVERSION_TYPE_TABX,
                }
                for i in range(nr):
                    kargs['val_{}'.format(i)] = conversion['param_val_{}'.format(i)]
                    kargs['text_{}'.format(i)] = conversion['text_{}'.format(i)]

                new_conversion = v4b.ChannelConversion(
                    **kargs
                )

                conversion = new_conversion

            elif conversion_type == v3c.CONVERSION_TYPE_RTABX:

                nr = conversion['ref_param_nr'] - 1
                kargs = {
                    'val_param_nr': nr * 2,
                    'ref_param_nr': nr + 1,
                    'conversion_type': v4c.CONVERSION_TYPE_RTABX,
                    'default_addr': conversion.referenced_blocks['default_addr']['text'],
                }
                for i in range(nr):
                    kargs['lower_{}'.format(i)] = conversion['lower_{}'.format(i)]
                    kargs['upper_{}'.format(i)] = conversion['upper_{}'.format(i)]
                    kargs['text_{}'.format(i)] = conversion.referenced_blocks['text_{}'.format(i)]['text']

                conversion = v4b.ChannelConversion(
                    **kargs
                )

            conversion.unit = unit

    return conversion
