const API_BASE = '/api/v1';

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    loadCollections();
    loadCollectionSelects();
    setInterval(loadStats, 5000); // Update stats every 5 seconds
});

// Tab management
function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Stats
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const stats = await response.json();
        
        document.getElementById('stats').innerHTML = `
            <div class="stat-item">
                <div class="stat-value">${stats.collections}</div>
                <div>Collections</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.total_documents}</div>
                <div>Documents</div>
            </div>
        `;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Collections
async function loadCollections() {
    try {
        const response = await fetch(`${API_BASE}/collections`);
        const collections = await response.json();
        
        const html = `
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Metric</th>
                        <th>Dimension</th>
                        <th>Documents</th>
                    </tr>
                </thead>
                <tbody>
                    ${collections.map(col => `
                        <tr>
                            <td>${col.name}</td>
                            <td>${col.metric}</td>
                            <td>${col.dimension}</td>
                            <td>${col.doc_count}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        document.getElementById('collections-list').innerHTML = html;
    } catch (error) {
        console.error('Error loading collections:', error);
    }
}

async function loadCollectionSelects() {
    try {
        const response = await fetch(`${API_BASE}/collections`);
        const collections = await response.json();
        
        const options = collections.map(col => 
            `<option value="${col.name}">${col.name}</option>`
        ).join('');
        
        document.getElementById('doc-collection-select').innerHTML = options;
        document.getElementById('search-collection-select').innerHTML = options;
    } catch (error) {
        console.error('Error loading collection selects:', error);
    }
}

function showCreateCollection() {
    document.getElementById('create-collection-modal').style.display = 'flex';
}

function hideCreateCollection() {
    document.getElementById('create-collection-modal').style.display = 'none';
}

async function createCollection() {
    const name = document.getElementById('collection-name').value;
    const metric = document.getElementById('collection-metric').value;
    const dimension = parseInt(document.getElementById('collection-dimension').value);
    
    try {
        const response = await fetch(`${API_BASE}/collections`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name,
                metric,
                dimension
            })
        });
        
        if (response.ok) {
            hideCreateCollection();
            loadCollections();
            loadCollectionSelects();
            alert('Collection created successfully!');
        } else {
            const error = await response.json();
            alert(`Error: ${error.error}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Search
async function performSearch() {
    const collection = document.getElementById('search-collection-select').value;
    const queryText = document.getElementById('search-query').value;
    const limit = parseInt(document.getElementById('search-limit').value);
    
    if (!collection || !queryText) {
        alert('Please select a collection and enter a query');
        return;
    }
    
    try {
        const query = queryText.split(',').map(x => parseFloat(x.trim()));
        
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                collection_name: collection,
                query,
                n: limit
            })
        });
        
        if (response.ok) {
            const results = await response.json();
            displaySearchResults(results);
        } else {
            const error = await response.json();
            alert(`Error: ${error.error}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function displaySearchResults(results) {
    const html = `
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Distance</th>
                    <th>Content</th>
                    <th>Metadata</th>
                </tr>
            </thead>
            <tbody>
                ${results.map(result => `
                    <tr>
                        <td>${result.id}</td>
                        <td>${result.distance.toFixed(4)}</td>
                        <td>${result.content}</td>
                        <td>${result.metadata}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    document.getElementById('search-results').innerHTML = html;
}

// Console
async function executeConsoleCommand() {
    const method = document.getElementById('console-method').value;
    const endpoint = document.getElementById('console-endpoint').value;
    const body = document.getElementById('console-body').value;
    
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (method === 'POST' && body) {
            options.body = body;
        }
        
        const response = await fetch(endpoint, options);
        const result = await response.json();
        
        document.getElementById('console-output').value = JSON.stringify(result, null, 2);
    } catch (error) {
        document.getElementById('console-output').value = `Error: ${error.message}`;
    }
}
