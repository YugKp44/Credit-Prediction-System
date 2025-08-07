import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

# Load the data
data = [
    {"Component": "Data Input", "Layer": "Input Layer", "Technology": "CSV/Database", "Position": 1, "Flow_to": "Data Processing"},
    {"Component": "Data Processing", "Layer": "Data Layer", "Technology": "Pandas/NumPy", "Position": 2, "Flow_to": "Feature Engineering"},
    {"Component": "Feature Engineering", "Layer": "Data Layer", "Technology": "Scikit-learn", "Position": 3, "Flow_to": "ML Model"},
    {"Component": "ML Model", "Layer": "Model Layer", "Technology": "Logistic Regression", "Position": 4, "Flow_to": "Prediction"},
    {"Component": "Prediction", "Layer": "Model Layer", "Technology": "Scikit-learn", "Position": 5, "Flow_to": "Credit Scoring"},
    {"Component": "Credit Scoring", "Layer": "Application Layer", "Technology": "Python", "Position": 6, "Flow_to": "Dashboard"},
    {"Component": "Dashboard", "Layer": "Application Layer", "Technology": "Streamlit", "Position": 7, "Flow_to": "User Interface"},
    {"Component": "User Interface", "Layer": "Presentation Layer", "Technology": "Streamlit/Plotly", "Position": 8, "Flow_to": "Database"},
    {"Component": "Database", "Layer": "Storage Layer", "Technology": "SQLite", "Position": 9, "Flow_to": "Data Input"}
]

df = pd.DataFrame(data)

# Define layer colors using the specified brand colors
layer_colors = {
    'Input Layer': '#1FB8CD',
    'Data Layer': '#DB4545', 
    'Model Layer': '#2E8B57',
    'Application Layer': '#5D878F',
    'Presentation Layer': '#D2BA4C',
    'Storage Layer': '#B4413C'
}

# Create positions for components in a circular flow layout
positions = {
    'Data Input': (2, 5),
    'Data Processing': (4, 5),
    'Feature Engineering': (6, 5),
    'ML Model': (8, 4),
    'Prediction': (8, 2),
    'Credit Scoring': (6, 1),
    'Dashboard': (4, 1),
    'User Interface': (2, 2),
    'Database': (2, 4)
}

# Create the figure
fig = go.Figure()

# Add components as scatter points with custom styling
for _, row in df.iterrows():
    x, y = positions[row['Component']]
    
    # Truncate component names and technology names to fit character limits
    comp_name = row['Component'][:10]
    tech_name = row['Technology'][:12]
    
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(
            size=80,
            color=layer_colors[row['Layer']],
            line=dict(width=3, color='white'),
            symbol='square'
        ),
        text=f"{comp_name}<br>{tech_name}",
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial Black'),
        name=row['Layer'],
        showlegend=True if row['Component'] == df[df['Layer'] == row['Layer']].iloc[0]['Component'] else False,
        hovertemplate=f"<b>{row['Component']}</b><br>Layer: {row['Layer']}<br>Tech: {row['Technology']}<extra></extra>"
    ))

# Add flow arrows between components
for _, row in df.iterrows():
    start_pos = positions[row['Component']]
    if row['Flow_to'] in positions:
        end_pos = positions[row['Flow_to']]
        
        # Calculate arrow direction
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Adjust start and end points to avoid overlap with markers
        factor = 0.15
        start_x = start_pos[0] + factor * dx
        start_y = start_pos[1] + factor * dy
        end_x = end_pos[0] - factor * dx
        end_y = end_pos[1] - factor * dy
        
        # Add arrow line
        fig.add_trace(go.Scatter(
            x=[start_x, end_x],
            y=[start_y, end_y],
            mode='lines',
            line=dict(width=3, color='rgba(70,70,70,0.8)'),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add arrow head
        arrow_x = end_x - 0.05 * dx
        arrow_y = end_y - 0.05 * dy
        
        fig.add_trace(go.Scatter(
            x=[arrow_x], y=[arrow_y],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=12,
                color='rgba(70,70,70,0.9)',
                line=dict(width=1, color='white')
            ),
            showlegend=False,
            hoverinfo='none'
        ))

# Update layout
fig.update_layout(
    title='Credit Risk System Architecture',
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 10]
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, 6]
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5,
        title_text="System Layers"
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Save the chart
fig.write_image('credit_risk_architecture.png')