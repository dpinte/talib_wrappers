""" Cython wrapper for the abstract TA-LIB library

This code allows to use the TA-Lib generic interface for calling all the TA
functions without knowing a priori the parameters. The Python layer adds some
convenience on top of the TA-Lib C code with easy introspection, type
validation, etc.

Author: Didrik Pinte <dpinte@enthought.com>

"""

# Python imports
import functools
import numpy as np

# Cython imports
cimport numpy
from cpython.string cimport PyString_AS_STRING


cdef extern from 'ta_libc.h':

    ctypedef int TA_RetCode
    ctypedef double TA_Real
    ctypedef int TA_Integer
    ctypedef unsigned int TA_FuncHandle
    ctypedef int TA_FuncFlags

    cdef enum:
        TA_SUCCESS

    cdef enum:
        TA_IN_PRICE_OPEN,
        TA_IN_PRICE_HIGH,
        TA_IN_PRICE_LOW,
        TA_IN_PRICE_CLOSE,
        TA_IN_PRICE_VOLUME,
        TA_IN_PRICE_OPENINTEREST,
        TA_IN_PRICE_TIMESTAMP,

    cdef enum:
        TA_OPTIN_IS_PERCENT,
        TA_OPTIN_IS_DEGREE,
        TA_OPTIN_IS_CURRENCY,
        TA_OPTIN_ADVANCED

    cdef enum:
        TA_OUT_LINE, TA_OUT_DOT_LINE

    ctypedef struct TA_StringTable:
        int size
        char** string

    ctypedef struct TA_FuncInfo:
        char* name
        char* group
        char* hint
        char* camelCaseName
        TA_FuncFlags flags

        unsigned int nbInput
        unsigned int nbOptInput
        unsigned int nbOutput

        TA_FuncHandle* handle

    ctypedef enum TA_InputParameterType:
        TA_Input_Price,
        TA_Input_Real,
        TA_Input_Integer

    ctypedef enum TA_OutputParameterType:
        TA_Output_Real,
        TA_Output_Integer

    ctypedef int TA_InputFlags
    ctypedef int TA_OutputFlags

    ctypedef struct TA_InputParameterInfo:
        TA_InputParameterType type
        char* paramName
        TA_InputFlags flags

    ctypedef struct TA_OutputParameterInfo:
        TA_OutputParameterType type
        char* paramName
        TA_OutputFlags flags

    ctypedef struct TA_RealRange:
        TA_Real  min
        TA_Real  max
        TA_Integer precision

    ctypedef struct TA_IntegerRange:
        TA_Integer  min
        TA_Integer  max

    ctypedef struct TA_RealDataPair:
        TA_Real     value
        char *string

    ctypedef struct TA_IntegerDataPair:
        TA_Integer  value
        char *string

    ctypedef struct TA_RealList:
        TA_RealDataPair *data
        unsigned int nbElement

    ctypedef struct TA_IntegerList:
        TA_IntegerDataPair *data
        unsigned int nbElement

    ctypedef enum TA_OptInputParameterType:
        TA_OptInput_RealRange,
        TA_OptInput_RealList,
        TA_OptInput_IntegerRange,
        TA_OptInput_IntegerList

    ctypedef int TA_OptInputFlags

    ctypedef struct TA_OptInputParameterInfo:
        TA_OptInputParameterType type
        char *paramName
        TA_OptInputFlags flags

        char *displayName
        void *dataSet
        TA_Real defaultValue
        char *hint
        char *helpFile

    ctypedef struct TA_RetCodeInfo:
        char *enumStr
        char *infoStr

    void TA_SetRetCodeInfo(TA_RetCode theRetCode, TA_RetCodeInfo *retCodeInfo )

    TA_RetCode TA_GroupTableAlloc( TA_StringTable **table )
    TA_RetCode TA_GroupTableFree ( TA_StringTable *table )

    TA_RetCode TA_FuncTableAlloc( char *group, TA_StringTable **table )
    TA_RetCode TA_FuncTableFree ( TA_StringTable *table )

    TA_RetCode TA_GetFuncHandle(char *name, TA_FuncHandle **handle )
    TA_RetCode TA_GetFuncInfo(TA_FuncHandle *handle, TA_FuncInfo **funcInfo )

    TA_RetCode TA_GetInputParameterInfo(
        TA_FuncHandle *handle, unsigned int paramIndex,
        TA_InputParameterInfo **info
    )
    TA_RetCode TA_GetOptInputParameterInfo(
        TA_FuncHandle *handle, unsigned int paramIndex,
        TA_OptInputParameterInfo **info
    )
    TA_RetCode TA_GetOutputParameterInfo(
        TA_FuncHandle *handle, unsigned int paramIndex,
        TA_OutputParameterInfo **info
    )

    ctypedef struct TA_ParamHolder:
        void *hiddenData

    TA_RetCode TA_ParamHolderAlloc(
        TA_FuncHandle *handle, TA_ParamHolder **allocatedParams
    )

    TA_RetCode TA_ParamHolderFree( TA_ParamHolder *params )
    TA_RetCode TA_SetInputParamIntegerPtr(
        TA_ParamHolder *params, unsigned int paramIndex, TA_Integer *value
    )

    TA_RetCode TA_SetInputParamRealPtr(
        TA_ParamHolder *params, unsigned int paramIndex,TA_Real *value
    )

    TA_RetCode TA_SetInputParamPricePtr(
        TA_ParamHolder *params, unsigned int paramIndex, TA_Real *open,
        TA_Real *high, TA_Real *low, TA_Real *close, TA_Real *volume,
        TA_Real *openInterest
    )

    TA_RetCode TA_SetOptInputParamInteger(
        TA_ParamHolder *params, unsigned int paramIndex,TA_Integer value
    )

    TA_RetCode TA_SetOptInputParamReal(
        TA_ParamHolder *params, unsigned int paramIndex,TA_Real value
    )

    TA_RetCode TA_SetOutputParamIntegerPtr(
        TA_ParamHolder *params, unsigned int paramIndex, TA_Integer *out
    )

    TA_RetCode TA_SetOutputParamRealPtr(
        TA_ParamHolder *params, unsigned int paramIndex, TA_Real *out
    )

    TA_RetCode TA_CallFunc(
        TA_ParamHolder *params, TA_Integer startIdx, TA_Integer endIdx,
        TA_Integer *outBegIdx, TA_Integer *outNbElement
    )


    TA_RetCode TA_Initialize()
    TA_RetCode TA_Shutdown()


# Parameter flags
OUT_LINE = TA_OUT_LINE


cdef inline check_status(TA_RetCode code, message='Error while executing call'):
    cdef TA_RetCodeInfo exception_info

    if code != TA_SUCCESS:

        TA_SetRetCodeInfo(code, &exception_info)
        raise ValueError(
            '{}\n{} : {}'.format(
                message, exception_info.enumStr, exception_info.infoStr
            )
        )


cpdef initialize():
    """ Initialize TA-LIB. This function must be called prior to any usage of
    the library.

    """

    cdef TA_RetCode return_code =  TA_Initialize()
    check_status(return_code, 'Cannot intialize TA-Lib')

# Always intialize the library when loading the module
initialize()


cpdef finalize():
    """ Free all the resources used by TA-Lib. This function must be called
    prior to exiting the application code.

    """


    cdef TA_RetCode return_code = TA_Shutdown()

    if return_code != TA_SUCCESS:
        raise Exception("Cannot finalize TA-Lib (%d)!\n" % return_code)

def function_groups():
    """ Return the list of names of the supported function groups available in
    TA-LIB.

    """

    cdef TA_StringTable *table
    cdef TA_RetCode retCode
    cdef int i

    results = []

    retCode = TA_GroupTableAlloc( &table )

    if ( retCode == TA_SUCCESS ):
        for i in range(table.size):
            results.append(table.string[i])

    TA_GroupTableFree( table )

    return results

def functions_in_group(str group=None):
    """ Return the names of all the function in the given group. """


    cdef TA_StringTable *table
    cdef TA_RetCode retCode
    cdef int i

    cdef char* c_group

    if group is not None:
        c_group = PyString_AS_STRING(group)
    else:
        c_group = 'NULL'

    retCode = TA_FuncTableAlloc(c_group, &table )

    results = []

    if (retCode == TA_SUCCESS ):
        for i in range(table.size):
            results.append(table.string[i])

    else:
        # FIXME: implement a valid error management
        print 'Error'

    TA_FuncTableFree( table )

    return results

cdef class InputParameter:

    cdef TA_InputParameterInfo* c_info

    cdef set_parameter(self, TA_InputParameterInfo* c_info):
        self.c_info = c_info

    property name:
        def __get__(self):
            return self.c_info.paramName

    property flags:
        def __get__(self):
            return self.c_info.flags

    property type:
        def __get__(self):
            if self.c_info.type == TA_Input_Price:
                return 'Price'
            elif self.c_info.type == TA_Input_Real:
                return 'Real'
            else:
                return 'Integer'

    def __str__(self):
        return '{} ({} - {})'.format(self.name, self.type, self.flags)

    cdef add_to_holder(self, TA_ParamHolder* holder, unsigned int index,
                       numpy.ndarray input_array):

        if self.c_info.type == TA_Input_Price:
            raise NotImplementedError()

        elif self.c_info.type == TA_Input_Real:
            TA_SetInputParamRealPtr(holder, index, <double*>input_array.data)
        else:
            TA_SetInputParamIntegerPtr(holder, index, <int*>input_array.data)

cdef class OptionalInputParameter:

    cdef TA_OptInputParameterInfo* c_info

    cdef set_parameter(self, TA_OptInputParameterInfo* c_info):
        self.c_info = c_info

    property name:
        def __get__(self):
            return self.c_info.paramName

    property display_name:
        def __get__(self):
            return self.c_info.displayName

    property default_value:
        def __get__(self):
            return self.c_info.defaultValue

    property hint:
        def __get__(self):
            return self.c_info.hint

    property help_file:
        def __get__(self):
            return self.c_info.helpFile

    property flags:
        def __get__(self):
            cdef TA_OptInputFlags flags = self.c_info.flags

            if flags == TA_OPTIN_IS_PERCENT:
                return 'percent'
            elif flags == TA_OPTIN_IS_DEGREE:
                return 'degree'
            elif flags == TA_OPTIN_IS_CURRENCY:
                return 'currency'
            elif flags == TA_OPTIN_ADVANCED:
                return 'advanced'
            else:
                return ''

    property type:
        def __get__(self):
            cdef void* dataset = self.c_info.dataSet

            if self.c_info.type == TA_OptInput_RealRange:
                return '[{} - {}] ({})'.format(
                    (<TA_RealRange*>dataset).min,
                    (<TA_RealRange*>dataset).max,
                    (<TA_RealRange*>dataset).precision
                )
            elif self.c_info.type == TA_OptInput_IntegerRange:
                return '[{} - {}]'.format(
                    (<TA_IntegerRange*>dataset).min,
                    (<TA_IntegerRange*>dataset).max,
                )
            elif self.c_info.type == TA_OptInput_IntegerList:
                integer_list = []
                element_count = (<TA_IntegerList*>dataset).nbElement
                for i in range(element_count):
                    value = (<TA_IntegerList*>dataset).data[i].value
                    name = (<TA_IntegerList*>dataset).data[i].string
                    integer_list.append( '{}:{}'.format(value, name))
                return  ','.join(integer_list)
            elif self.c_info.type == TA_OptInput_IntegerList:
                real_list = []
                element_count = (<TA_RealList*>dataset).nbElement
                for i in range(element_count):
                    value = (<TA_RealList*>dataset).data[i].value
                    name = (<TA_RealList*>dataset).data[i].string
                    real_list.append( '{}:{}'.format(value, name))
                return 'Real list: {}'.format( ','.join(real_list))
            else:
                raise NotImplementedError()

    def __str__(self):
        return '{} ({}{})'.format(self.name, self.type, self.flags)

    def __repr__(self):
        return self.__str__()

    cdef add_to_holder(self, TA_ParamHolder* holder, unsigned int index, value):

        cdef TA_RetCode return_code

        if self.c_info.type in [TA_OptInput_RealRange]:
            return_code = TA_SetOptInputParamReal(
                holder, index, <double> value
            )
        else:
            return_code = TA_SetOptInputParamInteger(
                holder, index, <int> value
            )

        check_status(return_code, 'Error while setting optional parameter!')

cdef class OutputParameter:

    cdef TA_OutputParameterInfo* c_info

    cdef set_parameter(self, TA_OutputParameterInfo* c_info):
        self.c_info = c_info

    property name:
        def __get__(self):
            return self.c_info.paramName

    property flags:
        def __get__(self):
            return self.c_info.flags

    property type:
        def __get__(self):
            if self.c_info.type == TA_Output_Real:
                return 'Real'
            else:
                return 'Integer'

    def __str__(self):
        return '{} ({} - {})'.format(self.name, self.type, self.flags)

    cdef numpy.ndarray add_to_holder(self, TA_ParamHolder* holder,
                                     unsigned int index, int size):

        cdef TA_RetCode return_code
        cdef numpy.ndarray output_ary

        if self.c_info.type == TA_Output_Real:
            output_ary = np.zeros( (size,), dtype=np.float64)
            return_code = TA_SetOutputParamRealPtr(
                holder, index, <TA_Real*>output_ary.data
            )
        else:
            output_ary = np.zeros( (size,), dtype=np.int32)
            return_code = TA_SetOutputParamIntegerPtr(
                holder, index, <int*>output_ary.data
            )

        check_status(return_code, 'Error while allocating output arrays!')

        return output_ary



cdef class TaFunction:

    cdef TA_FuncHandle* c_function
    cdef TA_FuncInfo* c_info

    def __cinit__(self, str function_name):

        cdef char* c_function_name = PyString_AS_STRING(function_name)
        cdef TA_RetCode retCode

        retCode = TA_GetFuncHandle(c_function_name, &self.c_function )

        if (retCode != TA_SUCCESS):

            raise RuntimeError(
                'Error while retrieving function handle - Error %d' % retCode
            )

        retCode = TA_GetFuncInfo(self.c_function, &self.c_info )

        if (retCode != TA_SUCCESS):

            raise RuntimeError(
                'Error while retrieving function info - Error %d' % retCode
            )


    property name:
        def __get__(self):
            return self.c_info.name

    property group:
        def __get__(self):
            return self.c_info.group

    property hint:
        def __get__(self):
            return self.c_info.hint

    property nb_input:
        def __get__(self):
            return self.c_info.nbInput

    property nb_optional_input:
        def __get__(self):
            return self.c_info.nbOptInput

    property nb_output:
        def __get__(self):
            return self.c_info.nbOutput

    property input_description:
        def __get__(self):
            result = []

            cdef TA_InputParameterInfo* param_info
            cdef InputParameter parameter
            cdef TA_Integer i

            for i in range(self.nb_input):

                retCode = TA_GetInputParameterInfo(
                    self.c_function, i, &param_info
                )
                if (retCode == TA_SUCCESS):
                    parameter = InputParameter()
                    parameter.set_parameter(param_info)
                    result.append(parameter)

            return result

    property optional_input_description:
        def __get__(self):
            result = []

            cdef TA_OptInputParameterInfo* param_info
            cdef OptionalInputParameter parameter
            cdef TA_Integer i

            for i in range(self.nb_optional_input):

                retCode = TA_GetOptInputParameterInfo(
                    self.c_function, i, &param_info
                )
                if (retCode == TA_SUCCESS):
                    parameter = OptionalInputParameter()
                    parameter.set_parameter(param_info)
                    result.append(parameter)

            return result


    property output_description:
        def __get__(self):
            result = []

            cdef TA_OutputParameterInfo* param_info
            cdef OutputParameter parameter
            cdef TA_Integer i

            for i in range(self.nb_input):

                retCode = TA_GetOutputParameterInfo(
                    self.c_function, i, &param_info
                )
                if (retCode == TA_SUCCESS):
                    parameter = OutputParameter()
                    parameter.set_parameter(param_info)
                    result.append(parameter)

            return result

    def __call__(self, start_index, end_index, *args, **kwargs):
        # check args
        if len(args) < self.nb_input:
            raise ValueError('Not enough inputs')
        if len(kwargs) > self.nb_optional_input:
            raise ValueError('Too many inputs')

        # create param holder
        cdef TA_ParamHolder* params
        cdef TA_RetCode return_code

        # allocate parameter holder
        return_code = TA_ParamHolderAlloc(self.c_function, &params)
        check_status(return_code, 'Cannot allocate param holder!')

        # create the input data pointers
        for index, input in enumerate(self.input_description):
            (<InputParameter?>input).add_to_holder(
                params, index, np.asarray(args[index])
            )

        # create the optional input data pointers

        for index, optional_input in enumerate(self.optional_input_description):
            param_name = optional_input.name
            if param_name in kwargs:
                (<OptionalInputParameter?>optional_input).add_to_holder(
                    params, index, kwargs[param_name]
                )

        # create the output holders
        cdef TA_Integer base_size = end_index - start_index + 1
        results = []
        for index, output in enumerate(self.output_description):
            results.append(
                (<OutputParameter?>output).add_to_holder(
                    params, <TA_Integer>index, base_size
                )
            )

        # call function
        cdef TA_Integer out_begin_index, out_nb_element

        return_code = TA_CallFunc(
            params, start_index, end_index, &out_begin_index, &out_nb_element
        )
        check_status(return_code, 'Error while calling TA function!')

        # dealloc parameter holder
        return_code = TA_ParamHolderFree(params)
        check_status(return_code, 'Cannot deallocate param holder!')

        return results, out_begin_index, out_nb_element
