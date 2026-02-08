import os
import matplotlib.pyplot as plt
import numpy as np
from neo4j import GraphDatabase

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://100.88.72.32:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neuroxiv")

# Query: Get pct_cells for Car3 subclass across all regions
QUERY_PANEL_B = """
MATCH (r:Region)-[hs:HAS_SUBCLASS]->(sc:Subclass)
WHERE sc.name = '001 CLA-EPd-CTX Car3 Glut'
RETURN r.acronym AS region, hs.pct_cells AS pct_cells
ORDER BY hs.pct_cells DESC
"""

# Color palette: Red (highest) -> Yellow (lowest) gradient
import matplotlib.colors as mcolors
EDGE_COLOR = '#2C3E50'  # Dark edge

def get_gradient_colors(n):
    """Generate red-to-yellow gradient colors for n bars."""
    red = np.array([231, 76, 60]) / 255    # #E74C3C
    yellow = np.array([241, 196, 15]) / 255  # #F1C40F
    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        rgb = red * (1 - t) + yellow * t
        colors.append(rgb)
    return colors


def fetch_panel_b_data():
    """Fetch region enrichment data from Neo4j KG."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        result = session.run(QUERY_PANEL_B)
        data = [(rec['region'], rec['pct_cells']) for rec in result]

    driver.close()
    return data


def plot_panel_b(data, output_path="outputs/ms_circuit/circuitB.png"):
    """Generate vertical bar chart for Panel B matching Panel E style."""
    regions = [d[0] for d in data]
    pct_values = [d[1] * 100 for d in data]  # Convert to percentage

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')

    # Vertical bar chart with red-to-yellow gradient
    x_pos = np.arange(len(regions))
    colors = get_gradient_colors(len(regions))
    bars = ax.bar(x_pos, pct_values, color=colors, edgecolor=EDGE_COLOR,
                  linewidth=0.8, width=0.7)

    # X-axis labels (rotated like Panel E)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions, rotation=45, ha='right', fontsize=13, fontweight='bold')

    # Y-axis
    ax.set_ylabel('Percentage of Cells (%)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(pct_values) * 1.15)  # Add headroom for labels

    # # Title
    # ax.set_title('B    Region Enrichment of Car3 Subclass (001 CLA-EPd-CTX Car3 Glut)',
    #              fontsize=14, fontweight='bold', loc='left')

    # X-axis label
    ax.set_xlabel('Brain Region', fontsize=16, fontweight='bold')

    # Add percentage labels on top of bars
    for bar, pct in zip(bars, pct_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, rotation=0)

    # Remove top and right spines (cleaner look like Panel E)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Light horizontal grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def main():
    print("Fetching Panel B data from Neo4j KG...")
    data = fetch_panel_b_data()
    print(f"Found {len(data)} regions with Car3 subclass")

    for region, pct in data:
        print(f"  {region}: {pct*100:.2f}%")

    print("\nGenerating Panel B figure...")
    output_path = plot_panel_b(data)
    print(f"\nDone! Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
