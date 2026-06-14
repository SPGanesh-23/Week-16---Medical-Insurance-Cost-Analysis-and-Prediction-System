import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

df = pd.read_csv('insurance.csv')

print('Shape:', df.shape)
print('\nColumn Names:')
print(df.columns)
df.head()

df.info()

print('Independent Variables (Features):')
print('  age      - Age of the primary beneficiary')
print('  sex      - Gender of the policyholder')
print('  bmi      - Body Mass Index')
print('  children - Number of dependents covered by the plan')
print('  smoker   - Smoking status')
print('  region   - Residential region in the US')
print()
print('Dependent Variable (Target):')
print('  charges  - Individual medical insurance cost billed by the insurer')

print('Gender Distribution:\n', df['sex'].value_counts(), '\n')
print('Smoker Distribution:\n', df['smoker'].value_counts(), '\n')
print('Region Distribution:\n', df['region'].value_counts())

print(df.isnull().sum())

print('Duplicate rows before cleaning:', df.duplicated().sum())

df = df.drop_duplicates()

print('Duplicate rows after cleaning: ', df.duplicated().sum())
print('Shape after removing duplicates:', df.shape)

le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex_encoded'] = le_sex.fit_transform(df['sex'])
df['smoker_encoded'] = le_smoker.fit_transform(df['smoker'])
df['region_encoded'] = le_region.fit_transform(df['region'])

print('sex    encoding:', dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_))))
print('smoker encoding:', dict(zip(le_smoker.classes_, le_smoker.transform(le_smoker.classes_))))
print('region encoding:', dict(zip(le_region.classes_, le_region.transform(le_region.classes_))))

df.head()

print(df.dtypes)

print('Average Insurance Charges ($)\n')
print(round(df['charges'].mean(), 2))

print('Average Age (years)\n')
print(round(df['age'].mean(), 2))

print('Average BMI\n')
print(round(df['bmi'].mean(), 2))

print('Average Number of Children\n')
print(round(df['children'].mean(), 2))

df[['age', 'bmi', 'children', 'charges']].describe()

smoker_avg = df.groupby('smoker')['charges'].mean().reset_index()
smoker_avg.columns = ['Smoker', 'Average Charges']
print(smoker_avg)

gender_avg = df.groupby('sex')['charges'].mean().reset_index()
gender_avg.columns = ['Gender', 'Average Charges']
print(gender_avg)

df['age_group'] = pd.cut(df['age'], bins=[17, 25, 35, 45, 55, 65],
                          labels=['18-25', '26-35', '36-45', '46-55', '56-65'])

age_group_avg = df.groupby('age_group')['charges'].mean().reset_index()
age_group_avg.columns = ['Age Group', 'Average Charges']
print(age_group_avg)

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

df['bmi_category'] = df['bmi'].apply(bmi_category)

bmi_avg = df.groupby('bmi_category')['charges'].mean().reset_index()
bmi_avg.columns = ['BMI Category', 'Average Charges']
print(bmi_avg)

segment_avg = df.groupby(['smoker', 'bmi_category'])['charges'].mean().reset_index()
segment_avg.columns = ['Smoker', 'BMI Category', 'Average Charges']
segment_avg = segment_avg.sort_values('Average Charges', ascending=False)
print(segment_avg)

top = segment_avg.iloc[0]
print('\nHighest-cost segment:')
print(f"  Smoker = {top['Smoker']}, BMI Category = {top['BMI Category']}, "
      f"Average Charges = ${top['Average Charges']:.2f}")

fig = px.bar(age_group_avg, x='Age Group', y='Average Charges',
              title='Average Insurance Charges by Age Group',
              text='Average Charges', color='Age Group',
              template='plotly_white')
fig.show()

fig = px.bar(bmi_avg, x='BMI Category', y='Average Charges',
              title='Average Insurance Charges by BMI Category',
              text='Average Charges', color='BMI Category',
              category_orders={'BMI Category': ['Underweight', 'Normal', 'Overweight', 'Obese']},
              template='plotly_white')
fig.show()

children_avg = df.groupby('children')['charges'].mean().reset_index()
children_avg.columns = ['Children', 'Average Charges']

fig = px.bar(children_avg, x='Children', y='Average Charges',
              title='Average Insurance Charges by Number of Children',
              text='Average Charges', template='plotly_white')
fig.show()

region_avg = df.groupby('region')['charges'].mean().reset_index()
region_avg.columns = ['Region', 'Average Charges']

fig = px.bar(region_avg, x='Region', y='Average Charges',
              title='Average Insurance Charges by Region',
              text='Average Charges', color='Region',
              template='plotly_white')
fig.show()

fig = px.bar(smoker_avg, x='Smoker', y='Average Charges',
              title='Average Insurance Charges: Smoker vs Non-Smoker',
              text='Average Charges', color='Smoker',
              color_discrete_map={'yes': 'red', 'no': 'green'},
              template='plotly_white')
fig.show()

fig = px.bar(gender_avg, x='Gender', y='Average Charges',
              title='Average Insurance Charges: Male vs Female',
              text='Average Charges', color='Gender',
              color_discrete_map={'male': 'steelblue', 'female': 'pink'},
              template='plotly_white')
fig.show()

corr_age = df['age'].corr(df['charges'])
print('Correlation between Age and Charges:', round(corr_age, 4))

corr_bmi = df['bmi'].corr(df['charges'])
print('Correlation between BMI and Charges:', round(corr_bmi, 4))

corr_children = df['children'].corr(df['charges'])
print('Correlation between Children and Charges:', round(corr_children, 4))

corr_smoker = df['smoker_encoded'].corr(df['charges'])
print('Correlation between Smoking Status and Charges:', round(corr_smoker, 4))

factors = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'charges']
corr_with_charges = df[factors].corr()['charges'].drop('charges')
ranked = corr_with_charges.reindex(corr_with_charges.abs().sort_values(ascending=False).index)

print('Factors ranked by strength of influence on Insurance Charges:\n')
print(ranked)

region_smoker_avg = df.groupby(['region', 'smoker'])['charges'].mean().reset_index()
region_smoker_avg.columns = ['Region', 'Smoker', 'Average Charges']

fig = px.bar(region_smoker_avg, x='Region', y='Average Charges', color='Smoker',
              barmode='group', title='Average Charges by Region and Smoking Status',
              color_discrete_map={'yes': 'red', 'no': 'green'}, template='plotly_white')
fig.show()

fig = px.histogram(df, x='age', nbins=20,
                    title='Age Distribution', template='plotly_white')
fig.show()

fig = px.histogram(df, x='bmi', nbins=20,
                    title='BMI Distribution', template='plotly_white')
fig.show()

fig = px.histogram(df, x='charges', nbins=30,
                    title='Insurance Charges Distribution', template='plotly_white')
fig.show()

fig = px.scatter(df, x='age', y='charges', color='smoker',
                  title='Age vs Insurance Charges',
                  color_discrete_map={'yes': 'red', 'no': 'green'},
                  template='plotly_white')
fig.show()

fig = px.scatter(df, x='bmi', y='charges', color='smoker',
                  title='BMI vs Insurance Charges',
                  color_discrete_map={'yes': 'red', 'no': 'green'},
                  template='plotly_white')
fig.show()

fig = px.box(df, x='smoker', y='charges', color='smoker',
              title='Insurance Charges by Smoking Status (Box Plot)',
              color_discrete_map={'yes': 'red', 'no': 'green'},
              template='plotly_white')
fig.show()

fig = px.box(df, x='region', y='charges', color='region',
              title='Insurance Charges by Region (Box Plot)',
              template='plotly_white')
fig.show()

corr_full = df[['age', 'bmi', 'children', 'sex_encoded',
                 'smoker_encoded', 'region_encoded', 'charges']].corr()

fig = px.imshow(corr_full, text_auto='.2f',
                 title='Correlation Heatmap',
                 color_continuous_scale='RdBu_r')
fig.show()

smoker_counts = df['smoker'].value_counts().reset_index()
smoker_counts.columns = ['Smoker', 'Count']

fig = px.pie(smoker_counts, names='Smoker', values='Count',
              title='Smoker vs Non-Smoker Distribution',
              color='Smoker', color_discrete_map={'yes': 'red', 'no': 'green'})
fig.show()

gender_counts = df['sex'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

fig = px.pie(gender_counts, names='Gender', values='Count',
              title='Gender Distribution',
              color='Gender', color_discrete_map={'male': 'steelblue', 'female': 'pink'})
fig.show()

low_thresh = df['charges'].quantile(0.33)
high_thresh = df['charges'].quantile(0.66)

def cost_category(charge):
    if charge <= low_thresh:
        return 'Low Cost'
    elif charge <= high_thresh:
        return 'Medium Cost'
    else:
        return 'High Cost'

df['Cost_Category'] = df['charges'].apply(cost_category)

print(f'Low-cost threshold  (33rd percentile): ${low_thresh:.2f}')
print(f'High-cost threshold (66th percentile): ${high_thresh:.2f}')
print()
print(df['Cost_Category'].value_counts())

cost_counts = df['Cost_Category'].value_counts().reset_index()
cost_counts.columns = ['Cost Category', 'Count']

fig = px.bar(cost_counts, x='Cost Category', y='Count',
              title='Customer Distribution by Cost Category',
              text='Count', color='Cost Category',
              category_orders={'Cost Category': ['Low Cost', 'Medium Cost', 'High Cost']},
              color_discrete_map={'Low Cost': 'green', 'Medium Cost': 'orange', 'High Cost': 'red'},
              template='plotly_white')
fig.show()

high_cost_df = df[df['Cost_Category'] == 'High Cost']

print('Number of High Cost customers:', len(high_cost_df))
print()
print('Smoking status among High Cost customers:')
print(high_cost_df['smoker'].value_counts())
print()
print('Average BMI of High Cost customers:', round(high_cost_df['bmi'].mean(), 2))
print('Average Age of High Cost customers:', round(high_cost_df['age'].mean(), 2))
print()
smoker_high_pct = (high_cost_df['smoker'] == 'yes').mean() * 100
print(f'{smoker_high_pct:.2f}% of High Cost customers are smokers')

X_simple = df[['age']]
y_simple = df['charges']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42)

print('Training set size:', X_train_s.shape)
print('Testing set size: ', X_test_s.shape)

lr_model = LinearRegression()
lr_model.fit(X_train_s, y_train_s)

print('Model trained successfully.')
print('Coefficient (slope):', round(lr_model.coef_[0], 2))
print('Intercept           :', round(lr_model.intercept_, 2))

y_pred_s = lr_model.predict(X_test_s)

comparison_s = pd.DataFrame({'Actual': y_test_s.values[:5], 'Predicted': y_pred_s[:5].round(2)})
print(comparison_s)

print(f"For every additional year of age, insurance charges increase by "
      f"approximately ${lr_model.coef_[0]:.2f} on average.")

line_data = X_test_s.sort_values('age')

fig = px.scatter(x=X_test_s['age'], y=y_test_s,
                  labels={'x': 'Age', 'y': 'Charges'},
                  title='Linear Regression: Age vs Insurance Charges',
                  template='plotly_white')
fig.add_scatter(x=line_data['age'], y=lr_model.predict(line_data),
                 mode='lines', name='Regression Line', line=dict(color='red'))
fig.show()

X_multi = df[['age', 'bmi', 'children', 'smoker_encoded']]
y_multi = df['charges']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

print('Training set size:', X_train_m.shape)
print('Testing set size: ', X_test_m.shape)

mlr_model = LinearRegression()
mlr_model.fit(X_train_m, y_train_m)

print('Model trained successfully.')
print('\nModel Coefficients:')
for feature, coef in zip(X_multi.columns, mlr_model.coef_):
    print(f'  {feature:<16}: {coef:>10.2f}')
print(f'  {"Intercept":<16}: {mlr_model.intercept_:>10.2f}')

y_pred_m = mlr_model.predict(X_test_m)

comparison_m = pd.DataFrame({'Actual': y_test_m.values[:5], 'Predicted': y_pred_m[:5].round(2)})
print(comparison_m)

r2_simple = r2_score(y_test_s, y_pred_s)
r2_multi = r2_score(y_test_m, y_pred_m)

print(f'Simple Linear Regression   (age only)                   R2 Score: {r2_simple:.4f}')
print(f'Multiple Linear Regression (age, bmi, children, smoker) R2 Score: {r2_multi:.4f}')
print()
print('The Multiple Linear Regression model explains significantly more variance')
print('in insurance charges, confirming that smoking status, BMI, and number of')
print('children are important additional predictors beyond age alone.')

median_charge = df['charges'].median()
df['insurance_category'] = (df['charges'] > median_charge).astype(int)

print('Median Insurance Charge: $', round(median_charge, 2))
print()
print(df['insurance_category'].value_counts())
print('0 = Low Cost Customer, 1 = High Cost Customer')

X_log = df[['age', 'bmi', 'children', 'smoker_encoded']]
y_log = df['insurance_category']

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42)

print('Training set size:', X_train_l.shape)
print('Testing set size: ', X_test_l.shape)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_l, y_train_l)

print('Model trained successfully.')

y_pred_l = log_model.predict(X_test_l)
y_prob_l = log_model.predict_proba(X_test_l)[:, 1]

print('Sample Predictions (1=High Cost, 0=Low Cost):', y_pred_l[:5])
print('High-Cost Probabilities:                    ', y_prob_l[:5].round(2))
print('Actual Values:                              ', y_test_l.values[:5])

high_risk_customers = X_test_l[y_pred_l == 1].copy()

print('Number of test customers predicted as High Cost:', len(high_risk_customers))
print()
print('Profile of customers predicted as High Cost (mean feature values):')
print(high_risk_customers.mean().round(2))

r2_s = r2_score(y_test_s, y_pred_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
rmse_s = np.sqrt(mse_s)

print('Linear Regression (Age -> Charges)')
print('R2 Score :', round(r2_s, 4))
print('MAE      :', round(mae_s, 2))
print('MSE      :', round(mse_s, 2))
print('RMSE     :', round(rmse_s, 2))

r2_m = r2_score(y_test_m, y_pred_m)
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
rmse_m = np.sqrt(mse_m)

print('Multiple Linear Regression (Age, BMI, Children, Smoker -> Charges)')
print('R2 Score :', round(r2_m, 4))
print('MAE      :', round(mae_m, 2))
print('MSE      :', round(mse_m, 2))
print('RMSE     :', round(rmse_m, 2))

comparison_df = pd.DataFrame({
    'Metric': ['R2 Score', 'MAE', 'MSE', 'RMSE'],
    'Linear Regression (Age only)': [round(r2_s, 4), round(mae_s, 2), round(mse_s, 2), round(rmse_s, 2)],
    'Multiple Linear Regression': [round(r2_m, 4), round(mae_m, 2), round(mse_m, 2), round(rmse_m, 2)]
})
comparison_df

accuracy = accuracy_score(y_test_l, y_pred_l)
precision = precision_score(y_test_l, y_pred_l)
recall = recall_score(y_test_l, y_pred_l)
f1 = f1_score(y_test_l, y_pred_l)

print('Accuracy  :', round(accuracy, 4))
print('Precision :', round(precision, 4))
print('Recall    :', round(recall, 4))
print('F1 Score  :', round(f1, 4))

cm = confusion_matrix(y_test_l, y_pred_l)

fig = px.imshow(cm,
                 labels=dict(x='Predicted', y='Actual', color='Count'),
                 x=['Low Cost', 'High Cost'],
                 y=['Low Cost', 'High Cost'],
                 text_auto=True,
                 title='Confusion Matrix - Logistic Regression',
                 color_continuous_scale='Blues')
fig.show()

print(classification_report(y_test_l, y_pred_l, target_names=['Low Cost', 'High Cost']))

from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

app.layout = html.Div([

    html.H1('Medical Insurance Cost Analysis Dashboard', style={'textAlign': 'center'}),

    html.Br(),

    html.Div([
        html.Div([
            html.Label('Age Range'),
            dcc.RangeSlider(
                id='age-slider',
                min=int(df['age'].min()), max=int(df['age'].max()),
                value=[int(df['age'].min()), int(df['age'].max())],
                marks={i: str(i) for i in range(18, 65, 5)}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Select Region'),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': r, 'value': r} for r in sorted(df['region'].unique())],
                value='All', clearable=False
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    html.Div([
        html.Div([
            html.Label('Select Gender'),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': g, 'value': g} for g in sorted(df['sex'].unique())],
                value='All', clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Select Smoker Status'),
            dcc.Dropdown(
                id='smoker-dropdown',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': s, 'value': s} for s in sorted(df['smoker'].unique())],
                value='All', clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label('Number of Children'),
            dcc.Dropdown(
                id='children-dropdown',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': str(c), 'value': c} for c in sorted(df['children'].unique())],
                value='All', clearable=False
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
    ]),

    html.Br(),

    dcc.Graph(id='charges-hist'),
    dcc.Graph(id='age-scatter'),
    dcc.Graph(id='bmi-scatter'),
    dcc.Graph(id='smoker-box'),
    dcc.Graph(id='region-bar'),
    dcc.Graph(id='gender-bar'),
    dcc.Graph(id='corr-heatmap'),
    dcc.Graph(id='cost-pie'),

])

@app.callback(
    [Output('charges-hist', 'figure'),
     Output('age-scatter', 'figure'),
     Output('bmi-scatter', 'figure'),
     Output('smoker-box', 'figure'),
     Output('region-bar', 'figure'),
     Output('gender-bar', 'figure'),
     Output('corr-heatmap', 'figure'),
     Output('cost-pie', 'figure')],

    [Input('age-slider', 'value'),
     Input('region-dropdown', 'value'),
     Input('gender-dropdown', 'value'),
     Input('smoker-dropdown', 'value'),
     Input('children-dropdown', 'value')]
)
def update_dashboard(age_range, selected_region, selected_gender, selected_smoker, selected_children):

    filtered_df = df.copy()

    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]

    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]

    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['sex'] == selected_gender]

    if selected_smoker != 'All':
        filtered_df = filtered_df[filtered_df['smoker'] == selected_smoker]

    if selected_children != 'All':
        filtered_df = filtered_df[filtered_df['children'] == selected_children]

    # 1. Histogram: Charges Distribution
    hist_fig = px.histogram(
        filtered_df, x='charges', nbins=30,
        title='Insurance Charges Distribution',
        template='plotly_dark'
    )

    # 2. Scatter: Age vs Charges
    age_fig = px.scatter(
        filtered_df, x='age', y='charges', color='smoker',
        title='Age vs Insurance Charges',
        color_discrete_map={'yes': 'red', 'no': 'green'},
        template='plotly_dark'
    )

    # 3. Scatter: BMI vs Charges
    bmi_fig = px.scatter(
        filtered_df, x='bmi', y='charges', color='smoker',
        title='BMI vs Insurance Charges',
        color_discrete_map={'yes': 'red', 'no': 'green'},
        template='plotly_dark'
    )

    # 4. Box: Smoker vs Non-Smoker
    box_fig = px.box(
        filtered_df, x='smoker', y='charges', color='smoker',
        title='Charges: Smoker vs Non-Smoker',
        color_discrete_map={'yes': 'red', 'no': 'green'},
        template='plotly_dark'
    )

    # 5. Bar: Region-wise Charges
    region_data = filtered_df.groupby('region')['charges'].mean().reset_index()
    bar_region_fig = px.bar(
        region_data, x='region', y='charges',
        title='Region-wise Average Charges',
        template='plotly_dark'
    )

    # 6. Bar: Gender-wise Charges
    gender_data = filtered_df.groupby('sex')['charges'].mean().reset_index()
    bar_gender_fig = px.bar(
        gender_data, x='sex', y='charges',
        title='Gender-wise Average Charges',
        template='plotly_dark'
    )

    # 7. Correlation Heatmap
    numeric_cols = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 'region_encoded', 'charges']
    corr_data = filtered_df[numeric_cols].corr()
    heatmap_fig = px.imshow(
        corr_data, text_auto='.2f',
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        template='plotly_dark'
    )

    # 8. Pie: Cost Category Distribution
    cost_data = filtered_df['Cost_Category'].value_counts().reset_index()
    cost_data.columns = ['Cost Category', 'Count']
    pie_fig = px.pie(
        cost_data, names='Cost Category', values='Count',
        title='Cost Category Distribution',
        template='plotly_dark'
    )

    return hist_fig, age_fig, bmi_fig, box_fig, bar_region_fig, bar_gender_fig, heatmap_fig, pie_fig


if __name__ == '__main__':
    app.run(debug=True)

print('=== Key Insights ===')
print()
print('1. Average Insurance Charges:')
print('   $', round(df['charges'].mean(), 2))
print()
print('2. Average Charges - Smoker vs Non-Smoker:')
print(df.groupby('smoker')['charges'].mean().round(2))
print()
print('3. Average Charges by BMI Category:')
print(df.groupby('bmi_category')['charges'].mean().round(2))
print()
print('4. Strongest Influencing Factors on Charges (by correlation):')
print(ranked.round(4))
print()
print('5. Regression Model Performance (R2 Score):')
print('   Simple Linear Regression (Age only)        :', round(r2_s, 4))
print('   Multiple Linear Regression (Age, BMI, etc.):', round(r2_m, 4))
print()
print('6. Logistic Regression Accuracy (High vs Low Cost):', round(accuracy * 100, 2), '%')
print()
print('Conclusion:')
print('Smoking status is the single strongest driver of medical insurance costs,')
print('followed by BMI and age. Multiple Linear Regression substantially')
print('outperforms Simple Linear Regression by incorporating these factors,')
print('and Logistic Regression effectively distinguishes high-cost customers')
print('to support more accurate premium estimation and risk-based policy planning.')