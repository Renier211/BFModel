# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 08:14:16 2022

@author: Enrico.Scheltema
"""
#%% 0.1 Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import time
from tqdm import tqdm
#import numba 
#from numba import njit
import warnings

#%% Triangle Operations Class

    
class TriangleOperations():
    """This Class hold all the Base operations performed on the triangle"""
    def __init__(self, originaltriangle):
        """
        Purpose
        -------
        Initialises class with cumulative triangle data

        Parameters
        ----------
        originaltriangle : pd.DataFrame
            Should be Cumulative traingle with loss quarters in rows (index) and
            columns are the transaction delays.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if isinstance(originaltriangle, pd.DataFrame):
            self.lossquarters = originaltriangle.index.tolist()
            self.basedata= originaltriangle.reset_index(drop=True).to_numpy()
            self.latest_diag_triangle = np.fliplr(np.tril(np.fliplr(self.basedata)))
            self.latest_diag_vector = np.diag(np.fliplr(self.basedata))
            
            if np.sum(self.basedata <0) > 0 :
                raise Exception('There are negative values in your cumulative triangle')
            
        else:
            raise Exception('Incorrect data types for one of the inputs')  
#------------------------------------------------------------------------------
    #@jit(nopython = True)            
    def dfs(self, inputTriangle):
        '''
        Purpose
        -------
        This function constructs Development factors and Gross development factors
        from input triangle. 
        
        Note the input triangle 

        Parameters
        ----------
        inputTriangle : numpy array
            DESCRIPTION.

        Returns
        -------
        DF - pandas dataframe of development factors
        GF - pandas dataframe of grossdevelopment factors

        '''
        
        
        DF = []
        n = inputTriangle.shape[0]
        #Get development factors and gross development factors
        for i in range(n-2):
            if np.sum(inputTriangle[0:(n-1-i), i]) != 0 :
                DF+=[np.sum(inputTriangle[0:(n-1-i), i+1])/np.sum(inputTriangle[0:(n-1-i), i])]
            else:
                DF+=[1]
        DF+=[1]
        
        #Convert to pd.Series
        #self.DFs_output = pd.DataFrame(DF , index = range(1,48), columns=["DFs"])
        DFs = DF
        
        DF+=[1]
        #Get Gross develop factors, by taking cumulative product
        #self.GDF_output = pd.DataFrame(np.flip(np.cumprod(np.flip(DF))), columns = ["GrossDFs"])
        GDF = np.flip(np.cumprod(np.flip(DF)))
        
        return np.array(DFs), GDF
#------------------------------------------------------------------------------
    @staticmethod
    def forward_project_triangle(cumulative_tri , df):
        """
        Purpose
        -------
        This function is used to construct the forward predicted cumulative
        triangle from the latest diagonals provided in the input cumulative 
        triangle. 
        
        It is a static method as use is outside of main processes. 

        Parameters
        ----------
        cumulative_tri : pd.DataFrame
            Input cumulative triangle.
        df : TYPE
            Input Development Factors corresponding to the cumulative triangle.

        Returns
        -------
        Forward projected cumulative triangle

        """
        if (isinstance(cumulative_tri , pd.DataFrame) and isinstance(df , np.ndarray)):
            output = cumulative_tri.copy()

            for i in range(output.shape[0] -1):
                #we take the cumulative product of each development factor, but 
                #ignore those not relevant anymore
                reverse_cumulative_product_DFs =  np.cumprod(df.copy()[i:])
                #apply factor
                output.iloc[-(i+1), i+1:] = output.iloc[-(i+1),i] * reverse_cumulative_product_DFs[0:len(reverse_cumulative_product_DFs)-1]
            
            return output
        else: 
            raise Exception("Incorrect input types")
            
#------------------------------------------------------------------------------
    #@jit(nopython = True)  
    def forwardProject(self, GDF):
        '''
        Purpose
        -------
        Projects forward the latest amounts with GDF. Note that the order should line
        in terms of how you want to apply the element wise product. Also give matrix
        of differences.

        Parameters
        ----------
        latest : numpy array
            array of latest observed (range from 0 to n).
        GDF : numpy array
            array of GDF (range frmo 0 to n).

        Returns
        -------
        projection

        '''
        latest = self.latest_diag_vector
        projections = np.multiply(latest, np.flip(GDF))
        differences = np.subtract(projections ,latest)
        
        return projections, differences
#------------------------------------------------------------------------------
    #@jit(nopython = True) 
    def residuals(self, data, standardise = False , standardised_DFs = None):
        '''
        Purpose
        -------
        Function produces triangle of bootrapped residuals

        Parameters
        ----------
        originaltriangle : numpy.array
            DESCRIPTION.
        standardise : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        
        data1 = np.array(data.copy())
        data2 = np.array(data.copy())
        
        data_2 = data2[:, 1:]
        #data_2 = data.copy()
        data_1 = data1[:, :-1]
        #data_1 = data.copy()
        residuals = np.divide(data_2,data_1, out=np.zeros_like(data_2), where=data_1!=0)
        
        
        if standardise == True:
            if isinstance(standardised_DFs, np.ndarray) == 0 :
                raise Exception('please first run dfs function before residuals and convert to pd.DataFrame')
            else:
                OriginalDFs = standardised_DFs[:-1]
                residuals = residuals / OriginalDFs.T
            
        self.originalresiduals = residuals
        
        return residuals
#------------------------------------------------------------------------------
    #@jit(nopython = True)  
    def bootstrap(self, residuals, seed_start = None , seed_iterator = None, method = "stratified" , sample_withreplacement = True ):
        '''
        Purpose
        -------
        This function will produce a bootrapped link ratio triangle. It will then 
        apply the residual going backword from the latest period to get a 
        bootstrapped cumulative triangle. 
        
        Methods of sampling include stratified (default) , parallelogram and
        completely random.

        Parameters
        ----------
        residuals : numpy array
            numpy array of link ratios.
            
        seed : int, optional
            If specified then the random sampling process will take sepecify input
            seed. The default is None.
            
        seed_iterator : int, optional
            This is the amount the seed will be increaed by in each run, to ensure
            different samples per run. Default is None

        method : str, optional
            Specifies the method of sampling (without replacement).
            - "stratified": Sampling takes place only within the respective column
                i.e. assume that transaction delays each have their own distribution
            - "parallelogram": This method makes a duplicate of the upper triangle
                and flips it around, so that a full row of link ratios are availible
                for each accident period sampling. 
                i.e. row 1 link ratios will be added to row n, and row 2 to row n-1
            -"random" sample accross the whole triangle
            
            The default is "stratified".   
        
        sample_withreplacement : int, optional
            This specifies if sampling should be done with (True) or without(false)
            replacement. Default is True

        Returns
        -------
        Bootstrapped cumulative development triangle as well as bootstrapped residuals.

        '''
        #0. Set random seed if provided
        if isinstance(seed_start,int):
            np.random.seed(seed_start + seed_iterator)
        
        #1.Sample without replacement per Columns (Stratified sampling as the 
        #   observed distribution of link ratios changes per transaction delay)
        res1 = residuals
        m = res1.shape[1]
        array = np.zeros((res1.shape[0], res1.shape[1]))
        
        if method == "stratified":
            for j in range(m):
                temp = res1[0:(m-j),j]
                resampled = np.random.choice(temp[:], temp.shape[0], replace=sample_withreplacement)
                array[0:(m-j),j] = resampled
        
        elif method == "parallelogram":
            #Make parallelogram by copying upper triangle matrix to lower triangle
            res_temp11 = res1.copy()
            res_temp22 = np.flipud(np.fliplr(res1.copy()))
            res_temp = np.add(res_temp11,res_temp22)   
            #Now resample over full rows:
            for j in range(m):
                temp = res_temp[0:m,j]
                resampled = np.random.choice(temp[:], temp.shape[0], replace=sample_withreplacement)
                array[0:(m),j] = resampled
       
        elif method =="random":
            #sample from accross the triangle
            #remove bottom row of zeros to make symmetrical
            res_temp = np.fliplr(res1.copy()[:-1,:])
            
            #get indeces for upper triangle
            uppertriangleindex = np.triu_indices(res_temp.shape[0])
            link_ratios_vec = res_temp[uppertriangleindex]
            #resample and construct new upper triangle matrix
            resampled = np.random.choice(link_ratios_vec[:], link_ratios_vec.shape[0], replace=sample_withreplacement)
            array[uppertriangleindex] = resampled
            array = np.fliplr(array)
            
       
        #2. Now we fill the lower diagonal of link ratios = 0s with 1s
        #   this won't actually affect algorithm, but helps us to use numpy's 
        #   cumulative product function that is applied from right to left per row
        res2 = np.fliplr(array)
        indeces_of_lower_triangle = np.tril_indices(res2.shape[0], k=-1)
        res2[indeces_of_lower_triangle]=1
        randomresid = np.fliplr(res2)
        #Add columns of 1s (which would also be 0s normally)
        #randomresid = pd.concat([pd.DataFrame(randomresid), pd.DataFrame(1, index = range(randomresid.shape[0]), columns =[randomresid.shape[1]] )] , axis=1)
       
        randomresid = np.concatenate( (randomresid, np.ones(randomresid.shape[0])[:,None]) , axis=1)
        
        #3. Get cumulative prod in reverse - multiplying link ratio from the latest
        #   observed going backwards - obtain reverse GDFs per row
        cumulativefactor = np.fliplr(np.apply_along_axis(np.cumprod, 1, np.fliplr(randomresid)))
        #   Invert the row wise GDFs in order to help us reconstruct the triangle in reverse
        ones = np.ones(cumulativefactor.shape)
        
        reverse_cumulativefactor = np.divide(ones, cumulativefactor)
        
        #   make a matrix full of latest values (in between step) 
        latest_matrix = np.tile(self.latest_diag_vector, (self.latest_diag_vector.shape[0],1)).T
        

        #4. Backfill our triangle using our inverse GDFs per row
        cumulativemat = np.multiply(latest_matrix, reverse_cumulativefactor)
        cumulativemat = np.fliplr(np.triu(np.fliplr(cumulativemat)))
        
        return randomresid, cumulativemat
    
#------------------------------------------------------------------------------
    def highlowmap(self, values = ["H","M","L"] ,  Threshold = 1 , standardise = False ,  standardised_DFs = None):
        """
        Purpose
        -------
        This function produces a High low map from the standardised residuals. 
        It uses the data in the background to create this mapping. The mapping
        is down with three specified levels, with the middle level centering 
        around the specified threshold.

        Parameters
        ----------
        values : list, optional
            Mapped values. Goes from largest to smalled. 
            The default is ["H","M","L"].
        Threshold : float, int , optional
            Corresponds to the middle value in the list. The default is 1.
        standardise : float, int , optional
            If true uses standardised residuals to construct heatmap. 
            The default is True.


        Returns
        -------
        Heatmap

        """
        if isinstance(values,list) and isinstance(Threshold,(int, float)):
            #standardise residuals
            data1 = np.array(self.basedata.copy())
            data2 = np.array(self.basedata.copy())
            
            data_2 = data2[:, 1:]
            #data_2 = data.copy()
            data_1 = data1[:, :-1]
            #data_1 = data.copy()
            residuals = np.divide(data_2,data_1, out=np.zeros_like(data_2), where=data_1!=0)
            
            if standardise:
                OriginalDFs = standardised_DFs[:-1]
                residuals = residuals / OriginalDFs.T
            
            heatmap = pd.DataFrame(np.nan, index = range(residuals.shape[0]) , columns = range(residuals.shape[1]))
            heatmap[residuals.copy() > Threshold] = 4
            heatmap[residuals.copy() == Threshold] = 3
            heatmap[residuals < Threshold ] = 2
            heatmap[residuals ==0 ] = 0
            for i in range(heatmap.shape[1]):
                heatmap.iloc[:,i] = heatmap.iloc[:,i].map({4:values[0] , 3:values[1] , 2: values[2] , 0:0 })
            
            self.heatmap = heatmap
            
            return self.heatmap
        
        else:
            raise Exception("Incorrect input types")
        
#------------------------------------------------------------------------------
    @staticmethod
    def yearquarter_list(end_yearquarter , start_yearquarter = "2011Q1" ):
        """
        Purpose
        -------
        This is a static function produces a list of year quarters from a given 
        start and end yearquarter in form YYYYQX e.g. "2021Q1"
        
        This is a static function and no output saved in background. It acts as 
        a helper function to contruct a triangle from unpivoted claims data. 

        Parameters
        ----------
        start_yearquarter : str (Optional)
            start yearquarter in form YYYYQX e.g. "2021Q1".
            Default is "2011Q1"
        end_yearquarter : str
            end yearquarter in form YYYYQX e.g. "2021Q1".

        Returns
        -------
        List of year month dates.

        """
        if isinstance(end_yearquarter, str) and isinstance(start_yearquarter, str):
            
            #Check
            check = (int(end_yearquarter[0:4]))*12 + (int(end_yearquarter[5]))*3 + 2 - (int(start_yearquarter[0:4]))*12 - (int(start_yearquarter[5]))*3
            if check <= 0 :
                raise Exception("Your start quarter must be less than your end quarter")
                
            #Get separate lists of the years and quarters
            difference =  (int(end_yearquarter[0:4]) + 1) - int(start_yearquarter[0:4]) 
            
            years = np.linspace(int(start_yearquarter[0:4]) , int(end_yearquarter[0:4])  , difference)
            quarters = ["Q1" , "Q2" , "Q3" , "Q4"]
            
            #Make a dataframe repeating all the years 4 times, then combine 
            #   with quarters list to get all possible year quarter combos
            Years = np.concatenate((years.reshape(-1,1),years.reshape(-1,1),years.reshape(-1,1),years.reshape(-1,1) ), axis=1)
            Years = pd.DataFrame(Years, columns = quarters)
            for i in range(4):
                Years[quarters[i]] = Years[quarters[i]].astype(int).astype(str) + quarters[i]
            
            #Make sure we aren't including more year quarters than we should 
            #   in last year
            list_of_years = list(Years.to_numpy().reshape(1,-1)[0])
            index = int(list_of_years.index(end_yearquarter))+1
            output = list_of_years[:index]
            
            index1 = int(output.index(start_yearquarter))
            output = output[index1:]
            
            return output
        else:
            raise Exception('Inputs must be strings')
#------------------------------------------------------------------------------            
    @staticmethod
    def create_triangle(data, index_heading , column_heading , values, start_yearquarter , end_yearquarter):
        """
        Purpose
        -------
        Creates incremental and cumulative triangles from unpivotted claims data
        containing the LossQ's and transaction delays. 
        
        Output is cumulative and incremental triangle in numpy array format.
        
        Note output matrix is symmetrical (so won't extend beyond transaction delay
        if your date selection does not make it symmetrical i.e. if more delays
        than LossQs)

        Parameters
        ----------
        data : pd.DataFrame
            Unpivotted input data, must contain LossQs, transation delay and values
        index_heading : str
            Columnheading used to make the rows.
        column_heading : str
            Columnheading used to make the column.
        values : str
            Columnheading for values used to create pivot.
        start_yearquarter : str
            start yearquarter in form YYYYQX e.g. "2021Q1".
        end_yearquarter : str
            end yearquarter in form YYYYQX e.g. "2021Q1".


        Returns
        -------
        Output is cumulative and incremental triangle in numpy array format.

        """
        if {isinstance(end_yearquarter, str) and isinstance(start_yearquarter, str)
            and isinstance(data, pd.DataFrame) and isinstance(index_heading, str) 
            and isinstance(values, str) and isinstance(column_heading, str) }:
            
            #data defensive copy
            data1 = data.copy()
            
            #List of year quarters
            final_years = TriangleOperations.yearquarter_list(end_yearquarter,start_yearquarter )
            
            #pivot data on LossQ and TransDelay, trim quarters before start quarter, and end quarter
            data1 = data1.pivot_table(index = [index_heading] , columns = [column_heading] , values = values , aggfunc=np.sum )
            #   Trim before start quarter
            index = int(data1.index.tolist().index(start_yearquarter))
            data1 = data1.iloc[index:, :]
            #   Trim after end quarter
            index = int(data1.index.tolist().index(end_yearquarter))
            data1 = data1.iloc[:(index+1), :]
            
            
            #Contruct a blank_array will all possible LossQ and Transdelay to 
            #   fill up any missing columns or rows
            blank_array = pd.DataFrame(0 , index = final_years , columns = range(0, len(final_years)))
            final_array = data1 + blank_array
            
            #Create incremental and cumulative triangles
            #   replace nan's with 0 to create incremental triangle
            final_array_incremental = final_array.fillna(0).to_numpy()
            #   take cumulative sum by row, then take upper triangle matrix
            #   so that lower triangle become 0 (but need to flip mat in operation)
            final_array_cumulative = np.fliplr(np.triu(np.fliplr(np.apply_along_axis(np.cumsum, 1, final_array_incremental ))))
            
            return pd.DataFrame(final_array_incremental , index = final_years) , pd.DataFrame(final_array_cumulative , index = final_years)
            
        else:
            raise Exception('Inputs must be strings')  
#------------------------------------------------------------------------------
#%% Premium Operations Class
class PremiumOperations():
    """
        Base class to help perform all the operations related to Premium. 
    """
    def __init__(self, premiumdata, coilist , datestart_column, dateend_column ):
        """
        Purpose
        -------
        Initialises class

        Parameters
        ----------
        premiumdata : pd.DataFrame
            Must be in a standard format, showing the premium per coi and the 
            start and end of the period for which the premium is shown. The number
            of months in between is also useful. 
            
        coilist : list
            list of coi column headings in data
        
        datestart_column : str
            Column name for start dates 
        
        dateend_column : str
            Column name for end dates 
        
        Returns
        -------
        None.

        """
        #1. Check data types of inputs
        if {isinstance(premiumdata, pd.DataFrame) and isinstance(coilist, list) and
            isinstance(datestart_column, str) and isinstance(dateend_column, str)}:
            
            #Assign attributes
            self.premium_basedata = premiumdata
            self.premium_coicolumnsnames = coilist
            self.premium_startdatecolumn = datestart_column
            self.premium_dateendcolumn = dateend_column
            
            #Some checks on data
            #   2. Check if negative premiums
            if all(self.premium_basedata[coilist] <0) ==False :
                warnings.warn('There are negative values in your premium input')    
            
            #   3. Check if input columns are dataset
            inputcolumns = coilist + [datestart_column] + [dateend_column]
            premiumdatacolumns = premiumdata.columns.tolist()
            if (set(inputcolumns).issubset(set(premiumdatacolumns))) == False:
                raise Exception('Input columns not in dataset')
            
        else:
            raise Exception('Incorrect file input type, use pandas DataFrame')
#------------------------------------------------------------------------------        
    @staticmethod
    def create_month_quarter_lookup(startquarter, endquarter, start_month):
        """
        Purpose
        -------
        Helper function to generate a lookup field between months and quarters

        Parameters
        ----------
        startquarter : str
            Start yearquarter in form YYYYQX e.g. "2021Q1".
        endquarter : str
            End yearquarter in form YYYYQX e.g. "2021Q1".
        start_month (inclusive) : str
            start month in format "February" (MMMM)

        Returns
        -------
        Numpy array of date mappings 

        """
        if {isinstance(startquarter, str) and isinstance(endquarter, str) and
            isinstance(start_month, str)}:
            
            
            #1. Make 3 copies of each quarter beneath each other
            listofquarters = np.array(TriangleOperations.yearquarter_list(endquarter , startquarter )).reshape(-1,1)
            Full_listofquarters = np.concatenate((listofquarters,listofquarters,listofquarters), axis = 1).reshape(-1,1)
            
            #2. Make the list of columns, 
            #   get the difference in month from start to end quarter
            diff_months = (int(endquarter[0:4]))*12 + (int(endquarter[5]))*3 + 2 - (int(startquarter[0:4]))*12 - (int(startquarter[5]))*3
            #   construct start and end dates
            start_date = pd.to_datetime("1" + " " + start_month + " " + str(int(startquarter[0:4])-1))
            end_date = start_date +  pd.DateOffset(months=diff_months )
            #   then make date range building from start date
            date_range = pd.date_range(start_date,end_date, freq='MS').tolist()
            
            #3. Join quarter and month lists together
            output = np.concatenate((np.array(date_range).reshape(-1,1) , Full_listofquarters) , axis=1)
            return output    
        
        else :
            raise Exception('Input types are incorrect')
#------------------------------------------------------------------------------       
    def premium_monthly_divide(self):
        """
        Purpose
        -------
        Creates a large tables specifying the monthly premium for each month
        from the start to end of the specified dates in the given table. 
        
        It references te input attributes and columns. 

        Returns
        -------
        Table of monthly premiums per coi (pd.DataFrame)

        """
        #copy of input data
        
        data =  self.premium_basedata.copy()
        n = data.shape[0]
        #start for loop over rows
        n = data.shape[0]
        for i in range(n):
            #get each month from start to end month
            date_range = pd.date_range(data[self.premium_startdatecolumn][i],data[self.premium_dateendcolumn][i], freq='MS').tolist()
            #Then get monthly premiums and repeat to fill up all of months in list
            m = len(date_range)
            monthly_premiums = (data[self.premium_coicolumnsnames].iloc[i,:] / m).to_numpy().reshape(1,-1)
            #   Concat monthly premiums with date_rage
            temp_output = np.concatenate( (np.array(date_range).reshape(-1,1),  ( np.repeat(monthly_premiums , m , axis = 0 ))) , axis =1)
            
            if i ==0:
                output = temp_output
            else:
                output = np.concatenate((output,temp_output ) , axis = 0)
        
        
        return pd.DataFrame(output , columns = ["Month"] + self.premium_coicolumnsnames)

#------------------------------------------------------------------------------
    @staticmethod
    def mapping_join(mappingcolumns, maindata, maindata_join_column , mapping_name):
        """
        Purpose
        -------
        Helper function that uses your month mapping to map quarters and pivots

        Parameters
        ----------
        mappingcolumns : np.ndarray
            Two column array, first column is your key and corresponds to the 
            value that need to be mapped. Second column is the mapped values.
        monthlypremiumdata : pd.DataFrame
            Dataframe we need to map values to .
        maindata_join_column : str
            column that needs to be mapped
        mapping_name : pd.DataFrame
            New column name.

        Returns
        -------
        Array with mapped column.

        """
        #create dictionary mapping column one to coumn two)
        map_dictionary = dict(zip(mappingcolumns[:,0], mappingcolumns[:,1]))
       
        #Create new mapped column
        maindata[mapping_name] = maindata[maindata_join_column].map(map_dictionary)
        
        return maindata
#------------------------------------------------------------------------------
#%% BF Model Class
class BF_Model(TriangleOperations, PremiumOperations):
    def __init__(self, originaltriangle, premiumdata ,  premium_values_column , premium_datestart_column, premium_dateend_column , start_month = "February"):
        """
        Purpose
        -------
        Sets up the background data used for object. 

        Parameters
        ----------
        originaltriangle : pd.DataFrame
            This is just a cumulative triangle (it needs to be symmetrical)
        premiumdata : pd.DataFrame
            DESCRIPTION.
        premium_values_column : str
            Name of columns in premium data that contains values
        premiumdata : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #We first initialise everything to be the same as our TriangleOperations Base class
        self.original_input_triangle = originaltriangle
        TriangleOperations.__init__(self, originaltriangle)
        PremiumOperations.__init__(self, premiumdata ,  [premium_values_column] , premium_datestart_column, premium_dateend_column)
        self.start_month = start_month 
#-------------------------------------------------------------------------------    
    def chainladderresult(self):
        '''
        Purpose
        -------
        Function produces the chain ladder results from original chain ladder
        data. This can be used to compare with the Excel calculated results as
        an accuracy check on the process.

        Returns
        -------
        None.

        '''
        
        # 1.Get DFs, GDFs from the input cumulative triangle (ignores lower triangle projected values)
        self.originalDF, self.originalGDF = super(BF_Model,self).dfs(self.basedata.copy())
        # 2.Use GDFs, Cumulative triangle and latest observed to project forward
        #    Obtain Ultimate and IBNR outstanding per period.
        self.Chainladder_Ultimate , self.Chainladder_IBNR_per_period =  super(BF_Model,self).forwardProject(self.originalGDF)
        
        #Create Table
        chainladdertable = pd.DataFrame(self.latest_diag_vector , index = self.lossquarters)
        chainladdertable['GDF'] = np.flip(self.originalGDF)
        chainladdertable['Percentage_Developed'] = 1 / chainladdertable['GDF']
        self.percentage_developed = chainladdertable['Percentage_Developed']
        chainladdertable['Ultimate'] = self.Chainladder_Ultimate
        chainladdertable['IBNR'] = self.Chainladder_IBNR_per_period
        
        return chainladdertable , np.sum(self.Chainladder_IBNR_per_period )
#-------------------------------------------------------------------------------
    def independentestimate(self , parameters_ULR  , threshold_developed = 0.90 , threshold_relevance_ULR_years = 4 , ULR_fixed = None):
        """
        Creates Independent Estimate
        
        Parameters
        ----------
        parameters_ULR : pd.DataFrame
            a simple DataFrame showing which quarters are included in ULR calculation.
            Must have indexes that match the loss quarters and indicate 1 for include
            and 0 for exclude.
        
        threshold_developed : float (Optional)
            Will only consider loss_quarters who are developed more than this percentage
            when calculating the ULR
            (default is 0.9)            
            
        threshold_relevance_ULR_years : float,int (Optional)
            Will only consider loss_quarters going back x amount of years 
            when calculating the ULR, due to relevance. 
            (default is 4)
        
        ULR_fixed : float or None (Optional)
            User can specify a fixed ULR (no calculation will be performed)
            (default is None)
            
        Returns
        -------
        Independent Estimate DataFrame
        """
        if {isinstance(parameters_ULR, pd.DataFrame) and isinstance(threshold_developed, float)
            and isinstance(threshold_relevance_ULR_years, (float,int)) and 
            (isinstance(ULR_fixed, float) or ULR_fixed  == None) }:
            
            # 0.Store all input parameters
            self.indp_parameters_ULR = parameters_ULR
            self.indp_threshold_developed = threshold_developed
            self.indp_threshold_relevance_ULR_years = threshold_relevance_ULR_years
            self.indp_ULR_fixed = ULR_fixed
            
            
            # 1.Make a table with Premiums grouped by Financial Quarter.
            #   Get premium monthly table
            premium_monthly_table = super(BF_Model,self).premium_monthly_divide()
            
            #   Create a mapping to quarters and apply to monthly table
            self.quartermapping = PremiumOperations.create_month_quarter_lookup(startquarter= self.lossquarters[0],
                                                                           endquarter = self.lossquarters[-1],
                                                                           start_month= self.start_month)
            self.monthly_premiums_table =  PremiumOperations.mapping_join(self.quartermapping, 
                                            premium_monthly_table, "Month", "Financial Quarter")
            
            #   Pivot on Financial Month
            self.quarterly_premium_table = pd.pivot_table(data =self.monthly_premiums_table
                                        , values = self.premium_coicolumnsnames, 
                                        index = ["Financial Quarter"], aggfunc=np.sum)
            
            # 2.Add to date Loss Ratio
            Independent_Estimate_Table = self.quarterly_premium_table
            Independent_Estimate_Table.columns = ['Premiums']
            Independent_Estimate_Table["To date LR"] = self.latest_diag_vector / Independent_Estimate_Table['Premiums']
            
            # 3.ULR calculation
            #   Case1: Ultimate Loss ratio is not fixed
            if ULR_fixed == None:
                # Preparation:
                #   First we add the percentage developed and Chainladder ultimate - will remove later
                if hasattr(self, "percentage_developed" ) and hasattr(self, "Chainladder_Ultimate" ) :
                    Independent_Estimate_Table["percentage_developed"] = getattr(self, "percentage_developed")
                    Independent_Estimate_Table["Chainladder_Ultimate"] = getattr(self, "Chainladder_Ultimate")
                    Independent_Estimate_Table["LatestObservedClaims"] = self.latest_diag_vector
                else:
                    self.originalDF, self.originalGDF = super(BF_Model,self).dfs(self.basedata.copy())
                    self.Chainladder_Ultimate , self.Chainladder_IBNR_per_period =  super(BF_Model,self).forwardProject(self.originalGDF)
                    Independent_Estimate_Table["percentage_developed"]  = 1 / np.flip(self.originalGDF)
                    Independent_Estimate_Table["LatestObservedClaims"] = self.latest_diag_vector
                    Independent_Estimate_Table["Chainladder_Ultimate"] = self.Chainladder_Ultimate
                    
                #   We also add the parameters manual removal of quarters
                parameters_ULR.columns = ["ManualULRspecification"]
                Independent_Estimate_Table = pd.concat([Independent_Estimate_Table , parameters_ULR ] , axis = 1 )
                
                #   Add relevance column
                m = threshold_relevance_ULR_years * 4 
                Independent_Estimate_Table["YearRelevanceInclusion"] = 0
                column_index = int(Independent_Estimate_Table.columns.tolist().index("YearRelevanceInclusion"))
                Independent_Estimate_Table.iloc[-m : , column_index] = 1
                
                # Remove unneeded columns
                #   Manual removal of quarters and periods with insufficient development and year relevance
                ULRcalcTable = Independent_Estimate_Table.copy()
                ULRcalcTable = ULRcalcTable[ULRcalcTable["ManualULRspecification"] != 0]
                ULRcalcTable = ULRcalcTable[ULRcalcTable["percentage_developed"] > threshold_developed]
                ULRcalcTable = ULRcalcTable[ULRcalcTable["YearRelevanceInclusion"] != 0]
                
                # Caculation:
                Independent_Estimate_Table["ULR"] =    np.sum(ULRcalcTable["Chainladder_Ultimate"]) / np.sum(ULRcalcTable["Premiums"])  
                    
            #   Case2: ULR is fixed
            else:
                Independent_Estimate_Table["ULR"] =  ULR_fixed
            
            # 4.Independent estimate Ultimate claims and IBRN
            Independent_Estimate_Table["Ultimate Claims"] =  Independent_Estimate_Table["Premiums"] * Independent_Estimate_Table["ULR"] 
            Independent_Estimate_Table["IBRN"] = Independent_Estimate_Table["Ultimate Claims"] - self.latest_diag_vector
        
            self.Independent_Estimate_Table = Independent_Estimate_Table  
            
            return Independent_Estimate_Table
        else:
            raise Exception('Error: Inputs of incorrect type')
        
#------------------------------------------------------------------------------
    def bf_estimate(self, chainladder,independentestimate):
        """
        Purpose
        -------
        Generate Bornheutter Ferguson estimate from the chainladder and independent
        estimate results. 
        
        To ensure that these two table have runned first. We specify them as 
        inputs, although column names etc. are assumed to be the same as specified
        in the chainladder and independent estimate columns separately. 
        
        Parameters
        ----------
        chainladder : pd.DataFrame
            Directly from chainladderresult function
        
        independentestimate : float (Optional)
            Directly from independentestimate function   
        
        Returns
        -------
        BF table with IBRN result

        """
        if {isinstance(chainladder, pd.DataFrame) 
            and isinstance(independentestimate, pd.DataFrame)}:
            
            #Check if all relevant columns in datasets
            columns1 = chainladder.columns.to_list()
            columns2 = independentestimate.columns.to_list()
            cl_columns = ["Percentage_Developed" , 'Ultimate']
            ie_columns = ["Premiums" , 'Ultimate Claims']
            
            #Case1: All relevant columns are present
            if (set(cl_columns).issubset(set(columns1)) and set(ie_columns).issubset(set(columns2))) :

                # Credibility factors
                BFTable = pd.DataFrame(chainladder["Percentage_Developed"])
                BFTable.columns = ["Credibility factor - CL"]
                BFTable["Credibility factor - IE"] = 1 - BFTable["Credibility factor - CL"]
                BFTable["Credibility factor - IE"][BFTable["Credibility factor - IE"]<0] = 0
                BFTable["BF UltimateClaims"] = BFTable["Credibility factor - CL"] * chainladder['Ultimate'] + BFTable["Credibility factor - IE"]* independentestimate["Ultimate Claims"]
                #Ultimate and BF
                BFTable["Implied Loss Ratio"] = BFTable["BF UltimateClaims"] /  independentestimate["Premiums"]
                BFTable["BF IBNR"] = BFTable["BF UltimateClaims"] - self.latest_diag_vector
                BFTable["BF IBNR Proportion"] = BFTable["BF IBNR"] / np.sum(BFTable["BF IBNR"])
                #Store attribute and return
                self.BF_table = BFTable
                return BFTable
            
            #Case2: Relevant columns are not present
            else:
                raise Exception("Please input datasets from chainladderresult and independentestimate functions")
        else:
            raise Exception('Error: Inputs of incorrect type')

#------------------------------------------------------------------------------
    def rollback_bf(self, rollback_period , period_of_test_increment , Graphs = True ):
        """
        Purpose
        ------
        This is a rollback method and will perform a full roll back analysis
        given the rollback_period and period of test increment. For this analysis
        to work you would have already needed to run 

        Parameters
        ----------
        rollback_period : int
            Number of quarters to rollback to. 
        period_of_test_increment : int
            Increments in the analysis. Must be a factor of rollback period
            (e.g. if rollback is 4 then increments of 1,2,4 apply).
            
        Graphs : bool (Optional)
            If True then accompanying graphs will be produced.
            Default is True
        
            

        Returns
        -------
        None.

        """
        if {isinstance(rollback_period, int) and isinstance(period_of_test_increment, int)
            and isinstance(Graphs, bool) }:
            
            if { (np.remainder(rollback_period , period_of_test_increment) != 0) 
                and (rollback_period >= period_of_test_increment)}:
                True
            else:
                raise Exception("Period of increments must be factor of rollback period")
            
            # 1.First get reduced triangle then run to get BF result
            reduced_triangle = self.original_input_triangle.copy()
            reduced_triangle = reduced_triangle.iloc[:-rollback_period, :-rollback_period]
            reduced_premium_data = self.indp_parameters_ULR.copy()
            reduced_premium_data = reduced_premium_data.iloc[:-rollback_period, :]
            
            # 2.Construct the BF result (chain ladder + Independent estimate)
            #   Initiate new BF_Model for rollback triangle
            rollback_Model = BF_Model(reduced_triangle.copy() , self.premium_basedata.copy() ,
                                      premium_values_column = self.premium_coicolumnsnames[0], 
                                      premium_datestart_column= self.premium_startdatecolumn,
                                      premium_dateend_column= self.premium_dateendcolumn,
                                      start_month= self.start_month)
            
            r_chaintable, r_chainIBRN = rollback_Model.chainladderresult()
            r_Independent_estimate = rollback_Model.independentestimate(reduced_premium_data, 
                                    threshold_developed = self.indp_threshold_developed,
                                    threshold_relevance_ULR_years =  self.indp_threshold_relevance_ULR_years ,
                                    ULR_fixed = self.indp_ULR_fixed )
            
            r_BF_result = rollback_Model.bf_estimate(r_chaintable, r_Independent_estimate)
            
            # 3.Now Create a rollback class, which will compare old and new tables
            main_cumulative_tri = TriangleOperations.forward_project_triangle(self.original_input_triangle.copy() , self.originalDF)
            r_cumulative_tri = TriangleOperations.forward_project_triangle(reduced_triangle.copy() , rollback_Model.originalDF)
            
            rollback_analysis  = BF_Model.Rollback(rollback_period , period_of_test_increment,
                                    r_BF_result , self.BF_table , r_cumulative_tri ,
                                    main_cumulative_tri)
            
            
            return rollback_analysis
            
        
        else:
            raise Exception('Error: Inputs of incorrect type')
        
        
            

#------------------------------------------------------------------------------
# Define an inner class Rollback - since you won't ever get a roll back without having a model first
    class Rollback:
        """
        This is a Rollback object used only when you have already initiated a 
        BF_model.
        """
        
        def __init__(self, rollback_period , period_of_test_increment , new_BF_Result , old_BF_Result , new_cumultriangle , old_cumultriangle ):
            """
            Parameters
            ----------
            rollback_period : int
                Number of quarters to rollback to. 
                
            period_of_test_increment : int
                Increments in the analysis (e.g. if rollback is 4 then increments of 1,2,4 apply). 
                
            new_BF_Result : pd.DataFrame
                The results table from your rollbacked model results 
            
            old_BF_Result : pd.DataFrame
                The results table from your Main model results 
                
            new_cumultriangle : pd.DataFrame
                The cumulative triangle from your rollbacked model results 
            
            old_cumultriangle : pd.DataFrame
                The cumulative triangle from your rollbacked model results  
                


            """
            if {isinstance(rollback_period, int) and isinstance(period_of_test_increment, int)
                and isinstance(new_BF_Result, pd.DataFrame) and isinstance(old_BF_Result, pd.DataFrame) 
                and isinstance(new_cumultriangle, np.ndarray) 
                and isinstance(old_cumultriangle, np.ndarray) }:
                
                self.rollback_period = rollback_period
                self.period_of_test_increment = period_of_test_increment
                self.new_BF_Result = new_BF_Result
                self.old_BF_Result = old_BF_Result
                self.new_cumultriangle = new_cumultriangle
                self.old_cumultriangle = old_cumultriangle

        
            else:
                raise Exception('Error: Inputs of incorrect type')
                
#------------------------------------------------------------------------------        
        def amount_analysis(self, graphs = True , displayGraphs = True, save_graph = False):
            """
            Purpose
            -------
            This function produces comparison tables between the actual and 
            the expected amounts. 
            
            The increments are done in relation to the period_of_test_increment
            parameter.
            
            Optional to produce graphs as well. 

            Parameters
            ----------
            graphs : bool, optional
                If true then produces graphs. The default is True.
            
            displayGraphs : bool, optional
                Turn off the display off graphs (although graphs will still be 
                returned via variable outputs). The default is True.

            Returns
            -------
            Summary tables and graphs stored in a list format. 

            """
            if {isinstance(graphs, bool) and isinstance(displayGraphs, bool) }:
                n = int(self.rollback_period / self.period_of_test_increment )
                
                #Get latest BF BE and latest diagonal
                Maintable =  pd.DataFrame(self.new_BF_Result["BF IBNR"])
                Maintable.columns = ["Total Best estimate"]
                Maintable["Latest at rollback"] = np.diag(np.fliplr(self.new_cumultriangle))
                
                Tables = []
                Tables1 = []
                c_tri_1 = self.new_cumultriangle.copy()
                c_tri_2 = self.old_cumultriangle.copy().iloc[:-self.rollback_period , : -self.rollback_period]
                
                for i in range(n):
                    if i ==0 :
                        #temp belongs to the latest values in interval, temp1 is previous
                        temp = Maintable.copy()
                        temp1 = pd.DataFrame(Maintable["Latest at rollback"])
                        temp1.columns = ["New observed start"]
                        temp1["Old observed start"] = np.diag(np.fliplr(c_tri_2))
                    
                    else:
                        temp = pd.DataFrame(Tables[i-1]["Remaining Best Estimate End"])
                        temp.columns = ["Remaining Best Estimate Start"]
                        temp1 = pd.DataFrame(Tables[i-1]["Old observed end"])
                        temp1.columns = ["Old observed start"]
                        temp1["New observed start"] = pd.DataFrame(Tables[i-1]["New observed end"])
                    
                    #contruct interval
                    m1 = i*self.period_of_test_increment
                    m2 = (i+1)*self.period_of_test_increment
                    
                    #mulitply add the Expected increment (new table cumutriangle)
                    tril11 = np.fliplr(np.tril(np.fliplr(c_tri_1.to_numpy()) , k=-1 - m1 ))
                    tril12 = np.fliplr(np.tril(np.fliplr(c_tri_1.to_numpy()) , k=-1 - m2)) 
                    temp["New observed end"] = np.sum(tril11 - tril12 , axis = 1)
                    temp["Expected Increment"] = temp["New observed end"] - temp1["New observed start"] 
                    
                    #add the Actual increment (old triangle)
                    tril21 = np.fliplr(np.tril(np.fliplr(c_tri_2.to_numpy()) , k=-1 - m1 ))
                    tril22 = np.fliplr(np.tril(np.fliplr(c_tri_2.to_numpy()) , k=-1 - m2)) 
                    temp["Old observed end"] = np.sum(tril21 - tril22 , axis = 1)
                    temp["Actual Increment"] = temp["Old observed end"] - temp1["Old observed start"]  
                    
                    #Get difference
                    temp["Difference (A-E)"] = temp["Actual Increment"] -  temp["Expected Increment"] 
                    
                    #Get Best estimate
                    if i ==0:
                        temp['Remaining Best Estimate End'] = temp["Total Best estimate"] - temp["Actual Increment"] 
                    else:
                        temp['Remaining Best Estimate End'] = temp["Remaining Best Estimate Start"] - temp["Actual Increment"] 
                    
                    #Store in list and go again. 
                    Tables += [temp]
                    Tables1 += [temp1]
                
                #format and arrange columns in output table
                for j in range(len(Tables)):
                    #just add the previous values used for calculation, to main table
                    temp2 = pd.concat([Tables[j] , Tables1[j] ] , axis = 1)
                    
                    #Order columns
                    if j ==0:
                        Tables[j] = temp2[["Total Best estimate","New observed start", "New observed end","Expected Increment", 
                                          "Old observed start","Old observed end", "Actual Increment", 
                                          "Difference (A-E)", 'Remaining Best Estimate End']]    
                    else:
                        Tables[j] = temp2[["Remaining Best Estimate Start", "New observed start", "New observed end","Expected Increment", 
                                          "Old observed start","Old observed end", "Actual Increment", 
                                          "Difference (A-E)", 'Remaining Best Estimate End']]
                if graphs:
                    n1 = len(Tables)
                    figs = []
                    
                    if displayGraphs:
                        plt.ioff()
                        
                    for i in range(n1):
                        #Get variables
                        Remaining_best_estim = Tables[i]['Remaining Best Estimate End']
                        Difference = Tables[i]["Difference (A-E)"]
                        m1 = i*self.period_of_test_increment
                        m2 = (i+1)*self.period_of_test_increment
                        title = "Development from Period " + str(m1) + " to " +str(m2)
                        #add figure
                        fig1 = plt.figure()
                        ax = fig1.add_subplot(111)
                        
                        #add are plot and bar plot
                        ax.fill_between(Remaining_best_estim.index.to_list(), Remaining_best_estim , alpha = 0.5 )
                        ax.bar(Difference.index.to_list(), Difference, align='center' , alpha = 0.5 )
                        #add plot parameters
                        ax.legend(["Remaining Best Estimate", "Difference (A - E)"])
                        ax.set_title(title)
                        ax.set_xlabel("loss quarters")
                        ax.set_ylabel("Rands")
                        
                        #Format x-axis, rotate lables 90 degrees and add every 3rd label. 
                        ax.tick_params(axis = 'x', rotation=90)
                        m = 3  # Keeps every 3rd label
                        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % m != 0]
                        
                        figs += [fig1]
                        if displayGraphs ==0:
                            plt.close()
                        
                    if displayGraphs ==0:
                        plt.ion()
                
                
                return Tables , figs                 
            
            else:
                raise Exception('Error: Inputs of incorrect type')

#------------------------------------------------------------------------------                
        def development_analysis(self, graphs = True , displayGraphs = True):
            """
            Purpose
            -------
            This function produces comparison tables between the actual and 
            the expected amounts. 
            
            The increments are done in relation to the period_of_test_increment
            parameter.
            
            Optional to produce graphs as well. 

            Parameters
            ----------
            graphs : bool, optional
                If true then will produce graphs. The default is True.
                
            displayGraphs : bool, optional
                Turn off the display off graphs (although graphs will still be 
                returned via variable outputs). The default is True.

            Returns
            -------
            Produces development analysis and also graphs. 

            """
            if {isinstance(graphs, bool) and isinstance(displayGraphs, bool) }:
                
                n = int(self.rollback_period / self.period_of_test_increment )
                
                #Get latest BF BE and latest diagonal
                Maintable =  pd.DataFrame(self.new_BF_Result["BF IBNR"])
                Maintable.columns = ["Total Best estimate"]
                Maintable["Latest at rollback"] = np.diag(np.fliplr(self.new_cumultriangle))
                
                Tables = []
                Tables1 = []
                c_tri_1 = self.new_cumultriangle.copy()
                c_tri_2 = self.old_cumultriangle.copy().iloc[:-self.rollback_period , : -self.rollback_period]
                
                for i in range(n):
                    if i ==0 :
                        temp = pd.DataFrame(Maintable["Latest at rollback"].copy())
                        temp1 = pd.DataFrame(Maintable["Latest at rollback"].copy())
                        temp1.columns = ["New observed start"]
                        temp1["Old observed start"] = np.diag(np.fliplr(c_tri_2))
                    
                    else:
                        temp = pd.DataFrame(Tables[i-1].copy())
                        temp1 = pd.DataFrame(Tables[i-1]["Old observed end"].copy())
                        temp1.columns = ["Old observed start"]
                        temp1["New observed start"] = pd.DataFrame(Tables[i-1]["New observed end"].copy())
                    
                    #contruct interval
                    m1 = i*self.period_of_test_increment
                    m2 = (i+1)*self.period_of_test_increment
                    

                    #add the Actual increment (old triangle)
                    tril11 = np.fliplr(np.tril(np.fliplr(c_tri_1.to_numpy()) , k=-1 - m1 ))
                    tril12 = np.fliplr(np.tril(np.fliplr(c_tri_1.to_numpy()) , k=-1 - m2)) 
                    temp["New observed end"] = np.sum(tril11 - tril12 , axis = 1)
                    temp["Expected Increment"] = ( temp["New observed end"] / temp1["New observed start"] ) 
                    
                    #add the Actual increment (old triangle)
                    tril21 = np.fliplr(np.tril(np.fliplr(c_tri_2.to_numpy()) , k=-1 - m1 ))
                    tril22 = np.fliplr(np.tril(np.fliplr(c_tri_2.to_numpy()) , k=-1 - m2)) 
                    temp["Old observed end"] = np.sum(tril21 - tril22 , axis = 1)
                    temp["Actual Increment"] = (temp["Old observed end"] / temp1["Old observed start"]) 
                    #Get difference
                    temp["Difference (A/E)"] = temp["Actual Increment"] /  temp["Expected Increment"]             
                   
                    Tables += [temp]
                    Tables1 += [temp1]
                    
                for j in range(len(Tables)):
                    temp2 = pd.concat([Tables[j] , Tables1[j] ] , axis = 1)
                    Tables[j] = temp2[["New observed start", "New observed end","Expected Increment", 
                                      "Old observed start","Old observed end", "Actual Increment", 
                                      "Difference (A/E)"]]
                if graphs:
                    if displayGraphs:
                        plt.ioff()
                    
                    n1 = len(Tables)
                    figs = []
                    for i in range(n1):
                        #Get variables
                        Difference = Tables[i]["Difference (A/E)"]
                        m1 = i*self.period_of_test_increment
                        m2 = (i+1)*self.period_of_test_increment
                        title = "Development from Period " + str(m1) + " to " +str(m2)
                        
                        #add figure
                        fig1 = plt.figure()
                        ax = fig1.add_subplot(111)
                        
                        #add are plot and bar plot
                        ax.plot(Difference.index.to_list(), Difference , marker = "o" , markerfacecolor = "#A68948" ,  alpha = 0.5 )
                        #add plot parameters
                        ax.legend(["Difference (A / E)"])
                        ax.set_title(title)
                        ax.set_xlabel("loss quarters")
                        ax.set_ylabel("Rands")
                        
                        #Format x-axis, rotate lables 90 degrees and add every 3rd label. 
                        ax.tick_params(axis = 'x', rotation=90)
                        m = 3  # Keeps every 3rd label
                        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % m != 0]
                        figs += [fig1]
                        if displayGraphs ==0:
                            plt.close()
                        
                    if displayGraphs ==0:
                        plt.ion()
                    
                return Tables , figs    
            else:
                raise Exception('Error: Inputs of incorrect type')    
        
                
#%% Risk Margin Class
class RiskMargin(TriangleOperations):
    def chainladderresult_original(self):
        '''
        Purpose
        -------
        Function produces the chain ladder results from original chain ladder
        data. This can be used to compare with the Excel calculated results as
        an accuracy check on the process.

        Returns
        -------
        None.

        '''
        
        # 1.Get DFs, GDFs from the input cumulative triangle (ignores lower triangle projected values)
        self.originalDF, self.originalGDF = super(RiskMargin,self).dfs(self.basedata.copy())
        # 2.Use GDFs, Cumulative triangle and latest observed to project forward
        #    Obtain Ultimate and IBNR outstanding per period.
        self.originalUltimate , self.originalIBNR_per_period =  super(RiskMargin,self).forwardProject(self.originalGDF)
        
        return np.sum(self.originalIBNR_per_period )
#------------------------------------------------------------------------------   

    
    def runriskmargin(self, n =10000, standardiseDFs = False , resid_boostrap_method = 'stratified', randomseed = None , save_run_no = None, sample_withreplacement = True):
        '''
        Purpose
        -------
        This function runs the main risk margin calculation, simulating n chain
        ladder reserving results and producing a 
        

        Parameters
        ----------
        n : int, optional
            Number of iterations in risk Margin. The default is 10000.
            
        standardiseDFs : TYPE, optional
            This specifies if we should standardise the link ratios beforehand
            by deviding by the DFs for the column they were in initially. All things
            being equal they appear to produce slightly smaller results. 
            
            The default is False.
            
        resid_boostrap_method : str, optional
            Specifies the method of sampling.
            - "stratified": Sampling takes place only within the respective column
                i.e. assume that transaction delays each have their own distribution
            - "parallelogram": This method makes a duplicate of the upper triangle
                and flips it around, so that a full row of link ratios are availible
                for each accident period sampling. 
                i.e. row 1 link ratios will be added to row n, and row 2 to row n-1
            -"random" sample accross the whole triangle
            
            The default is "stratified".   
        
            
        randomseed : int, optional
            Specify if a specific seed should be used. This will make the output
            for a given input the same each time the function runs. Note that
            the seed is used output on the first run, thereafter the seed is 
            incremented by 1 to produce each subsequent output (otherwise all runs
            would be exactly the same)
            
            The default is None.
            
        save_run_no : int, optional
            Specify which run to save. If nothing is selected then the last run
            (n) will be saved with the output. The default is None.
        sample_withreplacement : bool, optional
            Specify if sampling with(bootstrap) or without replacement.
            The default is True(with replacement).
            

        Returns
        -------
        Values from each chain ladder risk margin run.
        As well detailed stored calculations of a specified run / last run(default)

        '''
        #Just call function to get the orignal DFs in case we need to standardise
        self.originalDF, self.originalGDF = super(RiskMargin,self).dfs(self.basedata.copy())
        
        IBNR = []
        residuals = super(RiskMargin,self).residuals(self.basedata.copy() , standardise = standardiseDFs , standardised_DFs = self.originalDF)
        
        #Test if residuals less than or equal to 0:
        res_test = np.fliplr(residuals.copy())
        indeces_lowertriangle = np.tril_indices(res_test.shape[0], k=-1)
        res_test[indeces_lowertriangle] = 1
        if (res_test<=0).any():
            raise Exception("Negative or Zero value residuals factors detetected in main triangle")
        
        #Main Risk Margin loop
        for i in tqdm(range(n) , desc = "Iterations Progress Bar"):
            
            randomresid, bootstrap_cumul_triangle = super(RiskMargin,self).bootstrap(residuals, seed_start = randomseed , seed_iterator= i ,  method = resid_boostrap_method , sample_withreplacement=sample_withreplacement)
            DF, GDF = super(RiskMargin,self).dfs(bootstrap_cumul_triangle)
            Ultimate, IBNR_per_period = super(RiskMargin,self).forwardProject(GDF)
            IBNR+= [np.sum(IBNR_per_period)]
            
            if (isinstance(save_run_no,int) and save_run_no ==i): 
                #save specific run
                details_of_full_run = { 'bootstrap_cumul_triangle': bootstrap_cumul_triangle ,
                                       "original_resid" : residuals, 
                                       'randomresid' : np.fliplr(np.triu(np.fliplr(randomresid), k=1)) ,
                                       'DF' : DF[:-1] , 'GDF' : GDF ,
                                       'Ultimate': Ultimate , 'Latest_diagonals_vec': self.latest_diag_vector
                                       , 'IBNR_per_period' :IBNR_per_period }   
        if isinstance(save_run_no,int) ==0: 
            #otherwise save last no.
            details_of_full_run = { 'bootstrap_cumul_triangle': bootstrap_cumul_triangle ,
                                   "original_resid" : residuals, 
                                    'randomresid' : np.fliplr(np.triu(np.fliplr(randomresid), k=1)) ,
                                    'DF' : DF[:-1] , 'GDF' : GDF ,
                                    'Ultimate': Ultimate , 'Latest_diagonals_vec': self.latest_diag_vector
                                    , 'IBNR_per_period' :IBNR_per_period }   
    
        
        return pd.Series(IBNR), details_of_full_run

#-------------------------------------------------------------------------------
    @staticmethod    
    def risk_margin_convergence(IBNR_runs, upper_percentile = 0.75 , lower_percentile = 0.5, interval = 100,  graph = True ,custom_calc = False , custom_function = None , **graphkwaggs):
        '''
        Purpose
        -------
        This function generates a graph of risk margins. It is defaulted to take
        the difference of the upper and lower percentiles of the risk margins.
        
        The function cycles through all the risk margins iterations and applies
        the methodology to display the risk margin as it converges with increased
        iterations. 
        
        It is possible to implement your own function in the format:
            lamda x: function(x)  where x is your iterated list of IBNR values.

        Parameters
        ----------
        IBNR_runs : list
            Full list of IBNR runs.
        upper_percentile : float , optional
            Your upper percentile to use in risk margin calculation, default to 0.75
        lower_percentile : float , optional
            Your upper percentile to use in risk margin calculation, default to 0.5
        graph : bool , optional
            If true produces a graph of risk margins. 
        custom_calc : bool, optional
            If true then custom function is used for calulation. The default is False.
        custom_function : lambda function, optional
            Custom function to calculate risk margin, must be in above format
            hard code all other inputs. The default is None.

        Returns
        -------
        list of risk margins
        graph of risk margins

        '''
        RM_list = []
        N = np.ceil(len(IBNR_runs)/interval).astype('int')
        fig = plt.figure()
        ax = fig.add_subplot()
        if custom_calc == False:
            for i in range(N):
                if i< (N-1):
                    n = (i+1)*100
                    RM_list += [np.percentile(IBNR_runs[0:n] , upper_percentile*100 ) - np.percentile(IBNR_runs[0:n] , lower_percentile*100 )]
                elif i == (N-1):
                    RM_list += [np.percentile(IBNR_runs , upper_percentile*100 ) - np.percentile(IBNR_runs , lower_percentile*100 )] 
        else:
            for i in range(N):
                if i< (N-1):
                    n = (i+1)*100
                    RM_list += [custom_function(IBNR_runs[0:n])]
                elif i == (N-1):
                    RM_list += [custom_function(IBNR_runs)]
        if graph:
            N_10th = int(np.ceil(N*(0.9)))
            ax.plot(RM_list,**graphkwaggs)
            ax.axhline(y = np.average(RM_list[N_10th:]) , linestyle = '--' , color = 'red' , alpha = 0.5)
        try:
            RM_list = pd.Series(RM_list , index = range(interval,n +interval,interval) , name = 'RM_list per interval')
        except:
            RM_list =pd.Series(RM_list , name = 'RM_list per interval')
        
        return RM_list , fig , ax
#------------------------------------------------------------------------------




#%%Main:
if __name__ == "__main__":

#%% 0. Parameters and input data
    Coi = "Property"
    Cois = ["Motor" , "Property" , "Other" , "Liability"]
    position = Cois.index(Coi)
    
    #Parameters
    #   For Other 2012Q2 , For Liability "2012Q1" , Motor Propery "2011Q1"
    StartQuarter = ["2011Q1","2011Q1", "2012Q2" ,"2012Q1"][position]  
    EndQuarter = "2023Q1"
    FY_start_month = "March"
    # for Motor ["2021Q1" , "2021Q2"] , for the rest just []
    parameters_ULR_years = 4
    parameters_Exclude_ULR = [["2021Q1" , "2021Q2"] , [] , [] ,[]][position]  
    parameters_ULR_develpm = 0.9
    parameters_fix_ULR = None # Other scenario: 0.50
    
    #Input Data
    data_original = pd.read_excel("ClaimsInputData.xlsx", sheet_name="ClaimsData")
    premiums = pd.read_excel("PremiumInputData.xlsx", sheet_name="Input")    
    
#%% 1.Directly read in Triangle data (Alternative method)

    # #   Import Data
    # data = pd.read_excel("InputSheet.xlsx", sheet_name="Property")
    
    # #Drop unnecesary columns and values
    # OriginalTriangle = data.drop(['i','Loss_Q'], axis=1)
    
    # #Convert to numpy array, then flip column order, apply transformation to zero below main diagonal, flip back and convert to pandas
    # OriginalTriangle = pd.DataFrame(np.fliplr(np.triu(np.fliplr(OriginalTriangle.to_numpy()))))
    
    # #Set index to original - do at the end
    # OriginalTriangle= OriginalTriangle.set_index(data["Loss_Q"])    
    
#%% 1.1. Separate out data into cois and construct Triangles
    data1 = data_original.copy()
    data1 = data1[data1["COI_Map"] == Coi]
    # X is incremental triangle and y is cumulative triangle (numpy arrays.)
    x,y = TriangleOperations.create_triangle(data = data1, index_heading = "LossQ", column_heading ="TransDel",
                                             values = "Total", start_yearquarter = StartQuarter , end_yearquarter = EndQuarter)
    OriginalTriangle = y
    
#%% 1.2. Premium Data Input and BF Model Output
    paramtable = pd.DataFrame(1, index = OriginalTriangle.index.to_list() , columns = ["ULR incl/excl"])
    paramtable.loc[parameters_Exclude_ULR, :] = 0
    
    #Initialise model
    Model = BF_Model(OriginalTriangle , premiums , premium_values_column = Coi, 
                 premium_datestart_column= "Start" , premium_dateend_column= "End"
                 , start_month= FY_start_month)
    #Rum Chain Ladder, Independent Estimate and BF results
    chaintable, chainIBRN = Model.chainladderresult()
    Independent_estimate = Model.independentestimate(paramtable, threshold_developed = parameters_ULR_develpm,
                                                 threshold_relevance_ULR_years =  parameters_ULR_years ,
                                                 ULR_fixed = parameters_fix_ULR)
    BF_result = Model.bf_estimate(chaintable, Independent_estimate)
    Rollback_result= Model.rollback_bf(rollback_period = 4, period_of_test_increment =1)
    tab , figs = Rollback_result.amount_analysis(displayGraphs=True)
    tab1, figs = Rollback_result.development_analysis()
    #heatmap
    df,gdf = Model.dfs(Model.basedata)
    highlowmap = Model.highlowmap(standardise= True , standardised_DFs= df)
        
    
# %% 2.Run Normal Chain ladder as a check
    #2.1 Initialise class:
    RM_model = RiskMargin(OriginalTriangle)
    
    #2.2 Get data from main chainladder result (without bootstrap) 
    IBNR = RM_model.chainladderresult_original()
    print("The best estimate IBNR amount calculated by the best estimate is: R" + str(np.round(IBNR,2)))
    Original_data = {'link ratios_originals' : RM_model.residuals(RM_model.basedata.copy()) , 'DFs_originals' : RM_model.originalDF[:-1] , 
                     'GDFs_originals' :RM_model.originalGDF , 'Ultimate_originals':RM_model.originalUltimate , 
                     'latests_originals': RM_model.latest_diag_vector, 'IBNR per period_originals' : RM_model.originalIBNR_per_period}

# %% Run Bootstrap Chainladders    
    #2.3 Run different bootstrapped chain ladders - output is vector of all IBNR runs
    simulations_n = 10000
    #   Different Bootstrap options 
    #   (note seed is start seed and will iterate by 1 for each new triangle)
    IBNR2, rundetails2 = RM_model.runriskmargin(n=10000, standardiseDFs=True, resid_boostrap_method= 'stratified', save_run_no=999, randomseed= None, sample_withreplacement=True)
    IBNR1, rundetails1 = RM_model.runriskmargin(n=10000, standardiseDFs=False,  resid_boostrap_method= 'stratified', save_run_no=999, randomseed= None, sample_withreplacement=True)
    # IBNR1, rundetails1 = RM.runriskmargin(n=10000, standardiseDFs=False,  resid_boostrap_method= 'parallelogram' , sample_withreplacement=True)
    # IBNR1, rundetails1 = RM.runriskmargin(n=10000, standardiseDFs=False,  resid_boostrap_method= 'random')
    
    print("List of chain ladder simulation IBRN amounts \n" + str(np.round(IBNR2,2)))



# %% Calculate risk margins   (compares IBNR1 VS IBNR2) 
    #2.4 Riks Margin calculation
    RM1 = np.percentile(IBNR1, 75) - np.percentile(IBNR1, 50)
    RM2 = np.percentile(IBNR2, 75) - np.percentile(IBNR2, 50)
    RM = RM2
    
    print("The Risk Margin calculated from "+ str(simulations_n) + " is: R" + str(np.round(RM1,2)))
    print("The Risk Margin calculated from "+ str(simulations_n) + " is: R" + str(np.round(RM2,2)))

#Summary Table
    Summary_Table = pd.DataFrame([[np.sum(chaintable["IBNR"]) , RM , np.sum(chaintable["IBNR"]) + RM ] , 
                                  [np.sum(BF_result["BF IBNR"]) , RM , np.sum(BF_result["BF IBNR"]) +RM  ]] ,
                                  index = ["Chain Ladder" , "BF Model"],
                                  columns = ["Model IBRN" , "RM" , "Final" ])

#Add the IBRN to BF model Table and give a summary table of Total IBRN
    BF_result_final = BF_result.copy()    
    BF_result_final["RM"] = BF_result_final["BF IBNR Proportion"] * RM
    BF_result_final = pd.concat([BF_result_final , pd.DataFrame(np.sum(BF_result_final), columns = ["Total"]).T], axis=0)


# %%Convergence graphs
    Normal_implementation = IBNR1
    Standardisation = IBNR2
    interval = 100
    
    RMs = RiskMargin.risk_margin_convergence(Normal_implementation, graph = True, interval= interval)
    plt.title('Risk Margin Convergence Graph')
    plt.xlim([0, np.ceil(len(Normal_implementation)/interval)])
    plt.xlabel("Number of Interations (Interval size:" + str(interval) + ")")
    plt.ylabel("Risk Margin(R)")
    
    
    RMs, fig , ax = RiskMargin.risk_margin_convergence(Standardisation)
    #   Custom function version - where x is the input array of calculated IBNRs for each interval of iterations. 
    RMs_custom, fig , ax = RiskMargin.risk_margin_convergence(Normal_implementation, graph = True , custom_calc= True , custom_function= lambda x: np.percentile(x, 75) - np.percentile(x, 50))


#%% 3. Some additional plots and checks
    import WesternTools as WT
    
    graphs = WT.Distributions(pd.DataFrame(Standardisation, columns = ['values']))
    graphs.displot('values' , vertical_trendline = 0.0 ,  kind = "kde" )
    plt.title("Histogram of IBRN Risk margin run results")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    graphs = WT.Distributions(pd.DataFrame(Normal_implementation, columns = ['values']))
    additional = {"ax":ax}
    graphs.displot('values' )
    
