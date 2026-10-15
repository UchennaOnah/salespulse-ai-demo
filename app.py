from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).parent
REQUIRED=['order_id','date','product','category','region','quantity','unit_price']
st.set_page_config(page_title='SalesPulse AI 2.0',page_icon='📊',layout='wide')
st.markdown('''<style>
.stApp{background:#f4f7fb}.block-container{max-width:1450px;padding-top:1.4rem}
section[data-testid="stSidebar"]{background:#0b1831}section[data-testid="stSidebar"] *{color:#eef5ff}
.hero{background:linear-gradient(110deg,#0b1831,#174473 70%,#087e78);color:white;padding:30px;border-radius:18px;margin-bottom:20px}
.hero h1{color:white;margin:0;font-size:2.1rem}.hero p{color:#dbe8f6;margin:8px 0 0}
.kpi{background:white;border:1px solid #e5ebf4;border-radius:15px;padding:20px;min-height:128px;box-shadow:0 4px 18px #0b18310b}
.kpi-label{font-size:.85rem;color:#65748b}.kpi-value{font-size:1.8rem;font-weight:800;color:#14223b;margin-top:10px}
.insight{background:white;border-left:4px solid #14aa88;padding:15px 18px;border-radius:10px;margin:9px 0}
</style>''',unsafe_allow_html=True)
@st.cache_data
def sample(): return pd.read_csv(ROOT/'sample_sales.csv')
def validate(df):
    df=df.copy();df.columns=df.columns.str.strip().str.lower()
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    df['date']=pd.to_datetime(df['date'],errors='coerce')
    for c in ['quantity','unit_price']: df[c]=pd.to_numeric(df[c],errors='coerce')
    if df[REQUIRED].isna().any().any(): raise ValueError('Required values are missing or invalid')
    if (df.quantity<=0).any() or (df.unit_price<0).any(): raise ValueError('Invalid quantity or price')
    if df.order_id.astype(str).str.strip().eq('').any(): raise ValueError('Blank order ID')
    df['date']=df.date.dt.normalize();df['revenue']=df.quantity*df.unit_price
    return df
def features(dates,start):
    d=pd.DatetimeIndex(dates)
    return pd.DataFrame({'index':(d-start).days,'weekday':d.dayofweek,'month':d.month,'day':d.day,'weekend':(d.dayofweek>=5).astype(int)})
@st.cache_data(show_spinner=False)
def forecast(s,horizon):
    s=s.asfreq('D',fill_value=0)
    if len(s)<45:return None
    split=max(30,int(len(s)*.8));split=min(split,len(s)-7)
    x=features(s.index,s.index.min())
    model=RandomForestRegressor(n_estimators=140,min_samples_leaf=3,random_state=42,n_jobs=-1)
    model.fit(x.iloc[:split],s.iloc[:split]);pred=np.maximum(0,model.predict(x.iloc[split:]))
    mae=mean_absolute_error(s.iloc[split:],pred)
    baseline=mean_absolute_error(s.iloc[split:],np.repeat(s.iloc[:split].tail(7).mean(),len(s)-split))
    model.fit(x,s);dates=pd.date_range(s.index.max()+pd.Timedelta(days=1),periods=horizon)
    result=pd.DataFrame({'date':dates,'predicted_revenue':np.maximum(0,model.predict(features(dates,s.index.min())))})
    return result,mae,baseline,len(s)-split
def chart(fig):
    fig.update_layout(template='plotly_white',margin=dict(l=5,r=5,t=45,b=5),paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,use_container_width=True)
def card(label,value):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',unsafe_allow_html=True)
def insight(s):st.markdown(f'<div class="insight">{s}</div>',unsafe_allow_html=True)
def pounds(v):return f'£{v:,.0f}'
with st.sidebar:
    logo=base64.b64encode((ROOT/'assets/logo.svg').read_bytes()).decode()
    st.markdown(f'<img alt="SalesPulse AI logo" style="width:100%;margin-bottom:12px" src="data:image/svg+xml;base64,{logo}">',unsafe_allow_html=True)
    st.caption('BUSINESS INTELLIGENCE · V2.0')
    st.divider();uploaded=st.file_uploader('Upload sales CSV',type='csv')
    st.download_button('Download sample CSV',sample().to_csv(index=False).encode(),'sample_sales.csv','text/csv',use_container_width=True)
    st.caption('Columns: '+', '.join(REQUIRED))
    horizon=st.slider('Forecast horizon (days)',7,30,14)
try:df=validate(pd.read_csv(uploaded) if uploaded else sample())
except Exception as e:st.error(f'Unable to load data: {e}');st.stop()
st.markdown('<div class="hero"><h1>Grow smarter with SalesPulse AI</h1><p>Sales intelligence, business insights and explainable short-term forecasts.</p></div>',unsafe_allow_html=True)
st.info('SYNTHETIC DEMO DATA — not actual customer sales.' if not uploaded else 'Uploaded data — use only data you are authorised to process.')
a,b,c=st.columns([1,1,1.4])
with a:rs=st.multiselect('Regions',sorted(df.region.unique()),default=sorted(df.region.unique()))
with b:cs=st.multiselect('Categories',sorted(df.category.unique()),default=sorted(df.category.unique()))
with c:
    d=st.date_input('Date range',(df.date.min().date(),df.date.max().date()),min_value=df.date.min().date(),max_value=df.date.max().date())
if not isinstance(d,(list,tuple)) or len(d)!=2:st.info('Choose both dates');st.stop()
f=df[df.region.isin(rs)&df.category.isin(cs)&df.date.between(pd.Timestamp(d[0]),pd.Timestamp(d[1]))].copy()
if f.empty:st.warning('No data matches your filters');st.stop()
revenue=f.revenue.sum();orders=f.order_id.nunique();units=f.quantity.sum();aov=revenue/orders if orders else 0
for col,label,value in zip(st.columns(4),['TOTAL REVENUE','UNIQUE ORDERS','AVERAGE ORDER VALUE','UNITS SOLD'],[pounds(revenue),f'{orders:,}',f'£{aov:,.2f}',f'{units:,.0f}']):
    with col:card(label,value)
st.write('')
t1,t2,t3,t4,t5=st.tabs(['📈 Sales trends','🛍 Products & regions','💡 Business insights','🤖 AI forecast','📥 Reports & data'])
with t1:
    period=st.selectbox('Revenue aggregation',['Daily','Weekly','Monthly'],index=1)
    trend=f.set_index('date').revenue.resample({'Daily':'D','Weekly':'W','Monthly':'MS'}[period]).sum().reset_index()
    chart(px.area(trend,x='date',y='revenue',title='Revenue over time',color_discrete_sequence=['#287cf0']))
    st.dataframe(trend,use_container_width=True,hide_index=True)
with t2:
    left,right=st.columns(2)
    with left:
        reg=f.groupby('region',as_index=False).revenue.sum().sort_values('revenue',ascending=False)
        chart(px.bar(reg,x='region',y='revenue',title='Revenue by region',color='region'))
    with right:
        cat=f.groupby('category',as_index=False).revenue.sum()
        chart(px.pie(cat,names='category',values='revenue',hole=.55,title='Revenue by category'))
    st.subheader('Top products')
    top=f.groupby('product',as_index=False).agg(revenue=('revenue','sum'),units=('quantity','sum')).sort_values('revenue',ascending=False)
    st.dataframe(top.head(10),hide_index=True,use_container_width=True)
with t3:
    st.subheader('Automated business insights')
    for col,title in [('category','Category'),('region','Region'),('product','Product')]:
        s=f.groupby(col).revenue.sum().sort_values(ascending=False)
        insight(f'🏆 <b>{title} leader:</b> {s.index[0]} — {pounds(s.iloc[0])} ({s.iloc[0]/revenue:.1%} of revenue).')
    daily=f.groupby('date').revenue.sum().asfreq('D',fill_value=0)
    if len(daily)>=28:
        previous=daily.iloc[-28:-14].sum();current=daily.iloc[-14:].sum()
        if previous>0:insight(f'📊 <b>Recent momentum:</b> revenue changed {(current/previous-1):+.1%} over the last 14 days versus the preceding 14.')
    st.caption('These insights are rule-based summaries of the filtered dataset; they do not establish causes.')
with t4:
    st.subheader('Machine-learning revenue forecast')
    daily=f.groupby('date').revenue.sum().asfreq('D',fill_value=0)
    result=forecast(daily,horizon)
    if result is None:st.warning('Select at least 45 calendar days of sales history.')
    else:
        future,mae,base_mae,test_days=result
        q1,q2,q3=st.columns(3)
        q1.metric('Forecast total',pounds(future.predicted_revenue.sum()))
        q2.metric('Holdout MAE',pounds(mae))
        q3.metric('Baseline MAE',pounds(base_mae))
        fig=go.Figure();hist=daily.tail(90)
        fig.add_trace(go.Scatter(x=hist.index,y=hist.values,name='Historical revenue',line=dict(color='#287cf0')))
        fig.add_trace(go.Scatter(x=future.date,y=future.predicted_revenue,name='Forecast',line=dict(color='#12aa85',dash='dash')))
        fig.update_layout(title='Daily revenue: history and forecast',yaxis_title='Revenue (£)');chart(fig)
        st.caption(f'Holdout: last {test_days} historical days. MAE is average absolute daily error; lower is better. Baseline predicts the prior 7-day average. Forecasts are estimates, not guarantees.')
        if mae>=base_mae:st.warning('Model did not outperform the baseline on this holdout. Interpret with caution.')
        st.download_button('Download forecast CSV',future.to_csv(index=False).encode(),'salespulse_forecast.csv','text/csv')
with t5:
    st.subheader('Data explorer and exports')
    st.dataframe(f.sort_values('date',ascending=False).head(500),hide_index=True,use_container_width=True)
    st.download_button('Download filtered data',f.to_csv(index=False).encode(),'filtered_sales.csv','text/csv')
    report=pd.DataFrame([{'metric':'Total revenue','value':revenue},{'metric':'Unique orders','value':orders},{'metric':'Average order value','value':aov},{'metric':'Units sold','value':units}])
    st.download_button('Download KPI report',report.to_csv(index=False).encode(),'salespulse_kpi_report.csv','text/csv')
st.divider();st.caption('SalesPulse AI 2.0 · Demonstration edition · Forecasts are not financial advice. Do not upload sensitive client data to a public demo.')
