from urllib2 import urlopen
from xml.dom import minidom
import h5py
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import datetime
from scipy import stats,special
from math import sqrt
from functools import wraps
yqlURL="http://query.yahooapis.com/v1/public/yql?q="

#res = y.execute('select * from yahoo.finance.historicaldata where symbol = "goog"  and startDate="2010-10-01" and endDate="2010-10-30"')

symbol="^DJI"

startDate="2010-06-01"
endDate="2011-06-10"
dataFormat="&format=xml&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys"
callback="&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys"
historicalQ =yqlURL+"select%20*%20from%20yahoo.finance.historicaldata%20where%20symbol%20%3D%20%22"+ symbol \
             +"%22%20and%20startDate%20%3D%20%22"+ startDate +"%22%20and%20endDate%20%3D%20%22"+ endDate +"%22"+ dataFormat;

filename=symbol+startDate+endDate+".xml"

path=os.path.join("/home", "armen",  "Downloads")
os.chdir(path)

if not filename in os.listdir(path):

    u=urlopen(historicalQ)
    xmldata=u.read()
    try:
        
        xmlfile=open(filename,"w")
        xmlfile.write(xmldata)
        xmlfile.close()
    except IOError as (errno, strerror):
        print "I/O error({0}): {1}".format(errno, strerror)
xmldoc = minidom.parse(filename)
highestDayList=xmldoc.getElementsByTagName("High")
lowestDayList=xmldoc.getElementsByTagName("Low")
closeDayList=xmldoc.getElementsByTagName("Close")
datesList=xmldoc.getElementsByTagName("Date")

  
#class Symbol(IsDescription):
#    """This hold stock information for every day"""
#    name=StringCol(16)
#    highestDay=Float32Col()
#    lowestDay=Float32Col()
#    closeDay=Float32Col()

h5file = h5py.File("symbols.h5", "w")
symbolgroup = h5file.create_group("quotes")

#create a table named with the name of the symbol
dset=symbolgroup.create_dataset(symbol+"Prices",(len(highestDayList),3),'=f8')
dset.attrs["Columns"]="highestDay, lowestDay,closeDayList"

dsetdates=symbolgroup.create_dataset("symbolDates",(len(highestDayList),3),'i')
dsetdates.attrs["Columns"]="YYYY,MM,DD"



dates=[]
for k in range(len(highestDayList)):
    j=0
 
    for datepart in (datesList[-k-1].firstChild.data.split("-")):
        dsetdates[k-1,j]=int(datepart)
        j+=1
        
    dset[k-1,2]=float(highestDayList[-k-1].firstChild.data)
    dset[k-1,0]=float(lowestDayList[-k-1].firstChild.data)
    dset[k-1,1]=float(closeDayList[-k-1].firstChild.data)
    
    dates.append(datetime.date(dsetdates[k-1,0],dsetdates[k-1,1],dsetdates[k-1,2]))
    





class share(object):
    def __init__(self,share_prices,dates):
        self.share_prices=share_prices
        self.dates=dates
        
    def plotshare(self):
        datesaxis=mdates.date2num(self.dates)
        fig0=plt.figure()
        ax0=fig0.add_subplot(1,1,1)
        dateFmt = mdates.DateFormatter('%Y-%m-%d')
        ax0.xaxis.set_major_formatter(dateFmt)
        plt.minorticks_on()
        
        N=len(datesaxis)
        
        #ax0.xaxis.set_major_locator(DaysLoc)
        
        index=np.arange(N)
       # dev=np.abs(self.share_prices[:,0]-self.share_prices[:,2])
        
       # p0=plt.errorbar(index,self.share_prices,dev, fmt='.-',ecolor='green',elinewidth=0.1,linewidth=1)
        p0=plt.plot(index,self.share_prices)
        
        ax0.legend([p0],[symbol])
        ax0.set_ylabel( u'Index')
        ax0.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos=None: dates[int(x)]))
        ax0.set_xticks(np.arange(0,index[-1],4))
        ax0.set_xlim(index[0],index[-1])
        
        fig0.autofmt_xdate(rotation=90)
        fig0.savefig('./figures/sharesPrices.eps')
        plt.show()
    def returns(self):
        diff=np.diff(self.share_prices)
        dailyreturns=np.empty(diff.size)
        for index in range(diff.size):
            dailyreturns[index]=100*diff[index]/self.share_prices[index]
        return(returnSharevalues(dailyreturns))
  
    
class returnSharevalues(share):
    def __init__(self,returnsharevalues):
        self.return_share_values=returnsharevalues
        
    def plotinit(f):
        @wraps(f)
        def decorated_function(*args,**kwargs):
            fig0=plt.figure()
            ax0=fig0.add_subplot(1,1,1)
            return f(*args,**kwargs)
        return decorated_function
        
    def plotReturns(self):
        fig0=plt.figure()
        ax0=fig0.add_subplot(1,1,1)
        plt.plot(self.return_share_values,'r-+')
        plt.show()
        
    def plothist(self):
        fig2=plt.figure()
        ax0=fig2.add_subplot(1,1,1)
        Yrep=stats.norm.pdf(np.sort(self.return_share_values),np.mean(self.return_share_values),np.std(self.return_share_values))
        kdepdf=stats.gaussian_kde(self.return_share_values)
        ax0.plot(np.sort(self.return_share_values),Yrep, color="black")
        plt.hist(self.return_share_values,bins=40,normed=1)
        ax0.plot(np.sort(self.return_share_values),kdepdf.evaluate(np.sort(self.return_share_values)))
        fig2.savefig('./figures/histY.eps')
        plt.show()
        
    def plotcdf(self):
        fig1=plt.figure()
        ax0=fig1.add_subplot(1,1,1)

        F_Yrep=stats.norm.cdf(np.sort(self.return_share_values),np.mean(self.return_share_values),np.std(self.return_share_values))
        F_Y=np.arange(0.,1,1.0/len(self.return_share_values*1.0))
        p0=ax0.plot(np.sort(self.return_share_values),F_Y,'-')
        p1=ax0.plot(np.sort(self.return_share_values),F_Yrep,'--')
        fig1.savefig('./figures/CDFY.eps')
        plt.show()

    def qqplot(self):
        stats.probplot(self.return_share_values,sparams=(np.mean(self.return_share_values),np.std(self.return_share_values)),plot=plt)
     
    def samplepdfandplot(self):
        stdY=np.std(self.return_share_values)
        meanY=np.mean(self.return_share_values)
        samplesize=self.return_share_values.size
        Ysim=np.empty(samplesize)
        u=np.random.rand(samplesize)
        k=0
        for sample in u:
        #print meanY+stdY*sqrt(2)*special.erfinv(2*u[1]-1)
            Ysim[k]=meanY+stdY*sqrt(2)*special.erfinv(2*sample-1)
            k+=1
        fig3=plt.figure()
        fig3.add_subplot(1,1,1)
        plt.hist(Ysim,bins=20,normed=1)
        plt.show()
    
    def ploteverything(self):
        self.plotReturns()
        self.plothist()
        self.plotcdf()
        self.qqplot()
        self.samplepdfandplot()
            
myshare=share(dset.value[:-1,1],dates[1:])
myshare.plotshare()
myshare.returns().ploteverything()

def PosteriorParameters(Y):
    stdY=np.std(Y)
    meanY=np.mean(Y)
    rv=stats.t(Y.size-1,loc=meanY,scale=stdY**2/Y.size)
    
    #sigma_posterior=((Y.size-1)*stdY**2)/stats.chi2.pdf(Y[:-1],stdY**2)
    mu_posterior=stats.t.pdf(Y,Y.size-1)
    return(mu_posterior)







#plt.plot(np.sort(dset.value[:,1]),stats.norm.pdf(np.sort(dset.value[:,1]),np.mean(dset.value[:,1]),np.std(dset.value[:,1])))











#http://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20yahoo.finance.historicaldata%20where%20symbol\
#                                           %20%3D%20%22goog%22%20and%20startDate%20%3D%20%2222-10-2010%22%20and%20\
#                                           endDate%20%3D%20%2223-10-2010%22&format=json&env=store%3A%2F%2Fdatatables.org\
#                                           %2Falltableswithkeys
#http://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20yahoo.finance.historicaldata%20where%20symbol\
#                                           %20%3D%20%22goog%22%20and%20startDate%20%3D%20%222011-04-25%22%20and%20\
#                                           endDate%20%3D%20%222011-04-27%22&format=json&env=store%3A%2F%2Fdatatables.org\
#                                           %2Falltableswithkeys
#
