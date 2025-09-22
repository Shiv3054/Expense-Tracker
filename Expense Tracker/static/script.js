
document.documentElement.classList.add('js-loading');

document.addEventListener('DOMContentLoaded', () => {
    // Remove loading class after everything is fully loaded
    window.addEventListener('load', () => {
        setTimeout(() => {
            document.documentElement.classList.remove('js-loading');
        }, 100);
    });

    // Auth form handling
    const form = document.getElementById("auth-form");
    if (form) {
        handleAuthForm(form);
    }

    const isDashboardPage = document.getElementById("dashboard-container") !== null;
    if (isDashboardPage) {
        loadUserInfo();
        setTimeout(() => {
            updateDashboard();
        }, 100);

        // Add type selection change handler
        const typeSelect = document.getElementById('type');
        if (typeSelect) {
            typeSelect.addEventListener('change', handleTypeChange);
            // Initial call to set correct fields
            handleTypeChange();
        }

        // Manual transaction entry form
        const manualEntryForm = document.getElementById("manual-entry-form");
        if (manualEntryForm) {
            manualEntryForm.addEventListener("submit", handleManualEntry);
        }
    }

    // CSV file selection event listener
    const csvFileInput = document.getElementById('csvFile');
    if (csvFileInput) {
        csvFileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0] ? e.target.files[0].name : 'No file selected';
            document.getElementById('selectedFileName').textContent = fileName;
            
            if (e.target.files[0]) {
                previewCSV(e.target.files[0]);
            }
        });
    }

    // Upload CSV button event listener
    const uploadCSVBtn = document.getElementById('uploadCSVBtn');
    if (uploadCSVBtn) {
        uploadCSVBtn.addEventListener('click', handleCSVUpload);
    }
});

// Handle authentication form functionality
function handleAuthForm(form) {
    const formTitle = document.getElementById("form-title");
    const toggleText = document.getElementById("toggle-text");
    const message = document.getElementById("message");

    let isLogin = true;

    // Initialize the form state
    const setupInitialFormState = () => {
        const usernameInput = document.getElementById("signup-username");
        if (usernameInput) {
            usernameInput.style.display = isLogin ? "none" : "block";
        }
    };

    setupInitialFormState();

    // Toggle login/signup
    document.getElementById("toggle-text").addEventListener("click", (e) => {
        if (e.target.id === "toggle-link") {
            e.preventDefault();
            isLogin = !isLogin;

            formTitle.textContent = isLogin ? "Login" : "Signup";
            toggleText.innerHTML = isLogin
                ? `Don't have an account? <a href="#" id="toggle-link">Sign up</a>`
                : `Already have an account? <a href="#" id="toggle-link">Login</a>`;

            const usernameInput = document.getElementById("signup-username");
            if (usernameInput) {
                usernameInput.style.display = isLogin ? "none" : "block";
            }

            const submitButton = form.querySelector("button");
            if (submitButton) {
                submitButton.textContent = isLogin ? "Login" : "Signup";
            }

            if (message) {
                message.textContent = "";
            }
        }
    });

    // Handle form submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        if (!email || !password) {
            message.textContent = "Please fill in all required fields";
            return;
        }

        let data = { email, password };

        if (!isLogin) {
            const usernameInput = document.getElementById("signup-username");
            if (usernameInput) {
                const username = usernameInput.value;
                if (!username) {
                    message.textContent = "Please enter a username";
                    return;
                }
                data.username = username;
            }
        }

        const endpoint = isLogin ? "/login" : "/signup";

        try {
            message.textContent = "Processing...";

            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                if (isLogin) {
                    message.textContent = "Login successful! Redirecting...";
                    setTimeout(() => {
                        window.location.href = "/dashboard";
                    }, 500);
                } else {
                    message.textContent = "Signup successful! You can now log in.";
                    isLogin = true;
                    formTitle.textContent = "Login";
                    toggleText.innerHTML = `Don't have an account? <a href="#" id="toggle-link">Sign up</a>`;

                    const usernameInput = document.getElementById("signup-username");
                    if (usernameInput) {
                        usernameInput.style.display = "none";
                    }

                    const submitButton = form.querySelector("button");
                    if (submitButton) {
                        submitButton.textContent = "Login";
                    }

                    form.reset();
                }
            } else {
                message.textContent = result.error || "An error occurred";
            }
        } catch (error) {
            console.error("Auth error:", error);
            message.textContent = "A network error occurred. Please try again.";
        }
    });
}

// Function to preview CSV file contents
function previewCSV(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        Papa.parse(e.target.result, {
            header: true,
            skipEmptyLines: true,
            complete: function(results) {
                displayCSVPreview(results);
            },
            error: function(error) {
                showUploadStatus('Error parsing CSV: ' + error.message, 'danger');
            }
        });
    };
    reader.readAsText(file);
}

// Function to display CSV preview
function displayCSVPreview(results) {
    const previewContainer = document.getElementById('csvPreviewContainer');
    const previewBox = document.getElementById('previewBox');
    
    if (!previewContainer || !previewBox) return;
    
    // Check if we have data
    if (results.data && results.data.length > 0) {
        // Create table for preview
        let previewHTML = '<div class="table-responsive"><table class="table table-sm table-bordered">';
        
        // Headers
        previewHTML += '<thead><tr>';
        results.meta.fields.forEach(field => {
            previewHTML += `<th>${field}</th>`;
        });
        previewHTML += '</tr></thead>';
        
        // Data rows (show up to 5 rows)
        previewHTML += '<tbody>';
        const rowLimit = Math.min(results.data.length, 5);
        for (let i = 0; i < rowLimit; i++) {
            previewHTML += '<tr>';
            results.meta.fields.forEach(field => {
                previewHTML += `<td>${results.data[i][field] || ''}</td>`;
            });
            previewHTML += '</tr>';
        }
        previewHTML += '</tbody></table></div>';
        
        // Show preview with count info
        const countInfo = `<p>Showing ${rowLimit} of ${results.data.length} entries</p>`;
        previewBox.innerHTML = countInfo + previewHTML;
        previewContainer.style.display = 'block';
    } else {
        previewBox.innerHTML = '<p>No data found in the CSV file.</p>';
        previewContainer.style.display = 'block';
    }
}

// Function to handle the CSV upload
async function handleCSVUpload() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showUploadStatus('Please select a CSV file first.', 'warning');
        return;
    }
    
    if (!file.name.endsWith('.csv')) {
        showUploadStatus('Please upload a CSV file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showUploadStatus('Uploading and processing...', 'info');
        
        const response = await fetch('/upload_csv', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showUploadStatus(result.message, 'success');
            
            // Display category-wise transactions
            const previewContainer = document.getElementById('csvPreviewContainer');
            if (previewContainer && result.category_transactions) {
                let previewHTML = '<div class="category-transactions">';
                
                for (const [category, transactions] of Object.entries(result.category_transactions)) {
                    previewHTML += `
                        <div class="category-section">
                            <h4>${category}</h4>
                            <div class="table-responsive">
                                <table class="table table-sm table-bordered">
                                    <thead>
                                        <tr>
                                            <th>Type</th>
                                            <th>Amount</th>
                                            <th>Date</th>
                                            <th>Note</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                    `;
                    
                    transactions.forEach(txn => {
                        const amountClass = txn.type.toLowerCase() === 'income' ? 'income' : 'expense';
                        const amountSign = txn.type.toLowerCase() === 'income' ? '+' : '-';
                        previewHTML += `
                            <tr>
                                <td>${txn.type}</td>
                                <td class="${amountClass}">${amountSign}₹${txn.amount.toFixed(2)}</td>
                                <td>${formatDate(txn.date)}</td>
                                <td>${txn.note || ''}</td>
                            </tr>
                        `;
                    });
                    
                    previewHTML += `
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }
                
                previewHTML += '</div>';
                previewContainer.innerHTML = previewHTML;
                previewContainer.style.display = 'block';
            }
            
            // Update the entire dashboard
            await updateDashboard();
            
            // Clear the file input
            fileInput.value = '';
            
            // Clear the selected file name display
            const selectedFileName = document.getElementById('selectedFileName');
            if (selectedFileName) {
                selectedFileName.textContent = 'No file selected';
            }
        } else {
            showUploadStatus(result.error || 'Upload failed', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showUploadStatus('Upload failed: ' + error.message, 'error');
    }
}

async function handleManualEntry(e) {
    e.preventDefault();

    const type = document.getElementById('type').value;
    const amount = document.getElementById('amount').value;
    const note = document.getElementById('note')?.value || '';
    const categoryField = document.getElementById('category');
    const subcategoryField = document.getElementById('subcategory');

    // Toggle required attribute for category based on type
    if (type === 'Income') {
        categoryField.removeAttribute('required');
    } else {
        categoryField.setAttribute('required', 'true');
    }

    // Basic validation
    if (!type || !amount || (type === 'Expense' && !categoryField.value)) {
        showUploadStatus('Please fill in all required fields.', 'warning');
        return;
    }

    const formData = new FormData();
    formData.append('type', type);
    formData.append('amount', amount);
    formData.append('note', note);

    if (type === 'Income') {
        formData.append('category', 'Income');
        formData.append('subcategory', '');
    } else {
        formData.append('category', categoryField.value);
        formData.append('subcategory', subcategoryField.value || '');
    }

    formData.append('mode', 'Cash');

    try {
        const transactionData = {
            type,
            amount,
            note,
            category: type === 'Income' ? 'Income' : categoryField.value,
            subcategory: type === 'Income' ? '' : subcategoryField.value,
            mode: 'Cash'
        };
        console.log('Sending transaction data:', transactionData);

        const response = await fetch('/add_transaction', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log('Server response:', data);

        if (response.ok) {
            document.getElementById('manual-entry-form').reset();
            await updateDashboard();

            let message = 'Transaction added successfully!';
            if (data.totals) {
                message += `\nCurrent Totals:\nIncome: ₹${data.totals.income.toFixed(2)}\nExpenses: ₹${data.totals.expenses.toFixed(2)}\nNet Savings: ₹${data.totals.savings.toFixed(2)}`;
            }
            showUploadStatus(message, 'success');
            getBudgetAdvice();
        } else {
            showUploadStatus(`Error: ${data.error || 'Failed to add transaction'}`, 'danger');
        }
    } catch (error) {
        console.error('Error adding transaction:', error);
        showUploadStatus('Error: Failed to add transaction. Please try again.', 'danger');
    }
}

// Function to show upload status with appropriate styling
function showUploadStatus(message, type) {
    const statusElement = document.getElementById('uploadStatus');
    if (!statusElement) return;
    
    statusElement.textContent = message;
    statusElement.className = `alert alert-${type}`;
    statusElement.style.display = 'block';
    
    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            statusElement.style.display = 'none';
        }, 5000);
    }
}

// Function to handle manual transaction entry
document.addEventListener('DOMContentLoaded', () => {
    const manualForm = document.getElementById('manual-entry-form');
    const typeField = document.getElementById('type');
    const categoryField = document.getElementById('category');

    // Ensure the category field is only required when "Expense" is selected
    typeField.addEventListener('change', () => {
        if (typeField.value === 'Income') {
            categoryField.removeAttribute('required');
        } else {
            categoryField.setAttribute('required', 'true');
        }
    });

    manualForm.addEventListener('submit', handleManualEntry);
});

// Function to display imported transactions
function displayImportedTransactions(transactions) {
    // Get the transactions table body
    const tbody = document.querySelector('#transactionsTable tbody');
    if (!tbody) return;
    
    // Clear loading indicator if present
    const loadingRow = document.getElementById('loadingRow');
    if (loadingRow) {
        loadingRow.remove();
    }
    
    // If table is empty, clear any "no transactions" message
    if (tbody.innerText.includes('No transactions found')) {
        tbody.innerHTML = '';
    }
    
    // Add each transaction to the table
    transactions.forEach(txn => {
        // Create new row at the top of the table
        const newRow = tbody.insertRow(0);
        
        // Format the date
        let formattedDate = txn.date;
        try {
            // Try to parse and format the date (assuming ISO format)
            const date = new Date(txn.date);
            formattedDate = date.toLocaleDateString();
        } catch (e) {
            console.log('Could not format date:', e);
        }
        
        // Apply the appropriate class based on transaction type
        const amountClass = txn.type.toLowerCase() === 'income' ? 'income' : 'expense';
        
        // Format the amount with appropriate sign
        const formattedAmount = txn.type.toLowerCase() === 'income' 
            ? `+₹${parseFloat(txn.amount).toFixed(2)}`
            : `-₹${parseFloat(txn.amount).toFixed(2)}`;
        
        // Set row HTML
        newRow.innerHTML = `
            <td>${txn.type}</td>
            <td>${txn.category}</td>
            <td>${txn.subcategory || ''}</td>
            <td>${txn.note || ''}</td>
            <td class="${amountClass}">${formattedAmount}</td>
            <td>${formattedDate}</td>
        `;
        
        // Add a highlight effect to new rows
        newRow.classList.add('highlight-new');
        setTimeout(() => {
            newRow.classList.remove('highlight-new');
        }, 3000);
    });
    
    // Update any counters or totals
    updateTransactionStats();
}

// Load user info
async function loadUserInfo() {
    try {
        const response = await fetch("/user_info");

        if (response.ok) {
            const data = await response.json();
            const usernameDisplay = document.getElementById("user-name-display");
            if (usernameDisplay) {
                usernameDisplay.textContent = data.name || data.username || 'User';
            }
        } else {
            if (document.getElementById("dashboard-container")) {
                window.location.href = "/login";
            }
        }
    } catch (error) {
        console.error("Error loading user info:", error);
    }
}

// Function to load transactions from the server
async function loadTransactions() {
    const transactionsContainer = document.getElementById("transactions");
    if (!transactionsContainer) return;

    try {
        transactionsContainer.innerHTML = "<h3>Recent Transactions</h3><p>Loading transactions...</p>";

        const response = await fetch("/get_transactions");

        if (response.ok) {
            const data = await response.json();

            transactionsContainer.innerHTML = "<h3>Recent Transactions</h3>";

            if (!data.transactions || data.transactions.length === 0) {
                transactionsContainer.innerHTML += "<p>No transactions yet</p>";
                return;
            }

            const fragment = document.createDocumentFragment();

            data.transactions.forEach(txn => {
                const txnDiv = document.createElement("div");
                txnDiv.classList.add("transaction-entry");

                let content = `<strong>${txn.category}</strong>: $${parseFloat(txn.amount).toFixed(2)}`;

                if (txn.subcategory) {
                    content += ` (${txn.subcategory})`;
                }

                content += ` - ${formatDate(txn.date)}`;

                if (txn.note) {
                    content += `<br><em>${txn.note}</em>`;
                }

                txnDiv.innerHTML = content;
                fragment.appendChild(txnDiv);
            });

            transactionsContainer.appendChild(fragment);

            // Update budget advice if included in the response
            if (data.budget_advice) {
                const budgetAdviceContainer = document.getElementById('budget-advice');
                const adviceOutput = document.getElementById('advice-output');
                const formattedAdvice = data.budget_advice.replace(/\n/g, '<br>');
                
                if (budgetAdviceContainer) {
                    budgetAdviceContainer.innerHTML = formattedAdvice;
                }
                if (adviceOutput) {
                    adviceOutput.innerHTML = formattedAdvice;
                }
            }
        } else {
            transactionsContainer.innerHTML = "<h3>Recent Transactions</h3><p>Error loading transactions</p>";
        }
    } catch (error) {
        console.error("Error loading transactions:", error);
        transactionsContainer.innerHTML = "<h3>Recent Transactions</h3><p>Error loading transactions</p>";
    }
}

// Function to upload a receipt image
function uploadReceipt() {
    const fileInput = document.getElementById('receipt-file');
    const file = fileInput.files[0];
    
    if (!file) {
        showUploadStatus('Please select an image file first.', 'warning');
        return;
    }
    
    // Create FormData object
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading status
    showUploadStatus('Uploading receipt and processing with OCR...', 'info');
    
    fetch('/upload_receipt', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showUploadStatus('Receipt processed successfully!', 'success');
            // Refresh data
            loadTransactions();
            getBudgetAdvice();
            // Clear the file input
            fileInput.value = '';
        } else {
            showUploadStatus(`Error processing receipt: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        showUploadStatus(`Upload failed: ${error.message}`, 'danger');
    });
}

// Function to get budget advice
async function getBudgetAdvice() {
    const adviceContainer = document.getElementById('budgetAdvice');
    if (!adviceContainer) return;

    adviceContainer.innerHTML = 'Loading advice...';
    
    try {
        const response = await fetch('/get_budget_advice');
        if (!response.ok) {
            throw new Error('Failed to fetch budget advice');
        }
        
        const data = await response.json();
        
        if (data.advice) {
            // Replace newlines with <br> for proper display
            const formattedAdvice = data.advice.replace(/\n/g, '<br>');
            adviceContainer.innerHTML = formattedAdvice;
        } else {
            adviceContainer.innerHTML = 'No budget advice available. Please add more transactions.';
        }
    } catch (error) {
        console.error('Error getting budget advice:', error);
        adviceContainer.innerHTML = 'Error loading advice. Please try again.';
    }
}

// Function to handle logout
function logout() {
    // Simply redirect to logout endpoint which handles the session cleanup
    window.location.href = '/logout';
}

// Format date helper function
function formatDate(dateString) {
    if (!dateString) return "Unknown date";

    try {
        const date = new Date(dateString);
        return date.toLocaleDateString();
    } catch (e) {
        console.error("Error formatting date:", e);
        return "Invalid date";
    }
}

async function updateTransactionStats() {
    try {
        // Update spending insights
        const insightsContainer = document.getElementById('spending-insights');
        if (insightsContainer) {
            const response = await fetch('/get_spending_insights');
            if (response.ok) {
                const data = await response.json();
                if (data.insights && data.insights.length > 0) {
                    insightsContainer.innerHTML = data.insights.map(insight => 
                        `<div class="insight ${insight.type}">${insight.message}</div>`
                    ).join('');
                } else {
                    insightsContainer.innerHTML = '<p>No spending insights available yet.</p>';
                }
            }
        }
    } catch (error) {
        console.error('Error updating transaction stats:', error);
    }
}

// Function to update the entire dashboard
async function updateDashboard() {
    try {
        const response = await fetch("/get_transactions");
        if (!response.ok) {
            throw new Error('Failed to fetch transactions');
        }

        const data = await response.json();
        
        // Update transactions table
        const tbody = document.querySelector('#transactionsTable tbody');
        if (tbody) {
            tbody.innerHTML = ''; // Clear existing transactions

            if (!data.transactions || data.transactions.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center">No transactions found</td>
                    </tr>
                `;
            } else {
                data.transactions.forEach(txn => {
                    const row = document.createElement('tr');
                    
                    // Format the date
                    let formattedDate = txn.date;
                    try {
                        const date = new Date(txn.date);
                        formattedDate = date.toLocaleDateString();
                    } catch (e) {
                        console.log('Could not format date:', e);
                    }
                    
                    // Apply the appropriate class based on transaction type
                    const amountClass = txn.type === 'Income' ? 'income' : 'expense';
                    
                    // Format the amount with appropriate sign
                    const formattedAmount = txn.type === 'Income' 
                        ? `+₹${parseFloat(txn.amount).toFixed(2)}`
                        : `-₹${parseFloat(txn.amount).toFixed(2)}`;
                    
                    row.innerHTML = `
                        <td>${txn.type}</td>
                        <td>${txn.category}</td>
                        <td>${txn.subcategory || ''}</td>
                        <td>${txn.note || ''}</td>
                        <td class="${amountClass}">${formattedAmount}</td>
                        <td>${formattedDate}</td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }
        }

        // Update budget advice
        if (data.budget_advice) {
            const budgetAdviceContainer = document.getElementById('budgetAdvice');
            if (budgetAdviceContainer) {
                const formattedAdvice = data.budget_advice.replace(/\n/g, '<br>');
                budgetAdviceContainer.innerHTML = formattedAdvice;
                budgetAdviceContainer.style.display = 'block';
            }
        }

        // Update spending insights
        const insightsContainer = document.getElementById('spending-insights');
        if (insightsContainer) {
            const insightsResponse = await fetch('/get_spending_insights');
            if (insightsResponse.ok) {
                const insightsData = await insightsResponse.json();
                if (insightsData.insights && insightsData.insights.length > 0) {
                    insightsContainer.innerHTML = insightsData.insights.map(insight => 
                        `<div class="insight ${insight.type}">${insight.message}</div>`
                    ).join('');
                } else {
                    insightsContainer.innerHTML = '<p>No spending insights available yet.</p>';
                }
            }
        }
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}
// Function to handle type selection change
function handleTypeChange() {
    const typeSelect = document.getElementById('type');
    const categoryGroup = document.getElementById('category-group');
    const subcategoryGroup = document.getElementById('subcategory-group');
    
    if (!typeSelect || !categoryGroup || !subcategoryGroup) return;
    
    if (typeSelect.value === 'Income') {
        categoryGroup.style.display = 'none';
        subcategoryGroup.style.display = 'none';
    } else {
        categoryGroup.style.display = 'block';
        subcategoryGroup.style.display = 'block';
    }
}

async function getSpendingInsights() {
    const insightsContainer = document.getElementById('spending-insights');
    if (!insightsContainer) return;

    insightsContainer.innerHTML = 'Loading insights...';

    try {
        const response = await fetch('/get_spending_insights');
        if (!response.ok) {
            throw new Error('Failed to fetch spending insights');
        }
        const data = await response.json();
        if (data.insights && data.insights.length > 0) {
            insightsContainer.innerHTML = data.insights.map(insight =>
                `<div class="insight ${insight.type}">${insight.message}</div>`
            ).join('');
        } else {
            insightsContainer.innerHTML = '<p>No spending insights available yet.</p>';
        }
    } catch (error) {
        insightsContainer.innerHTML = 'Error loading insights. Please try again.';
    }
}

// Function to handle category selection change
async function handleCategoryChange(event) {
    const category = event.target.value;
    const categoryAdvice = document.getElementById('categoryAdvice');
    const mainTransactionsSection = document.getElementById('mainTransactionsSection');

    try {
        if (category === 'all') {
            if (categoryAdvice) categoryAdvice.style.display = 'none';
            if (mainTransactionsSection) mainTransactionsSection.style.display = 'block';
            await loadTransactions();
            return;
        }

        // Show loading state
        if (categoryAdvice) {
            categoryAdvice.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div><p>Loading category analysis...</p></div>';
            categoryAdvice.style.display = 'block';
        }

        // Get category advice
        const adviceResponse = await fetch(`/get_category_advice?category=${encodeURIComponent(category)}`);
        const adviceData = await adviceResponse.json();

        if (!adviceResponse.ok) {
            throw new Error(adviceData.error || 'Failed to get category advice');
        }

        // Calculate percentage based on current spending and total income
        const percentageUsed = adviceData.total_income > 0 
            ? ((adviceData.current_spent / adviceData.total_income) * 100).toFixed(1) 
            : 0;

        const warningThreshold = adviceData.warning_threshold || 75;
        const criticalThreshold = adviceData.critical_threshold || 90;

        // Update category advice section
        if (categoryAdvice) {
            categoryAdvice.innerHTML = `
                <div class="card mb-4">
                    <div class="card-body">
                        <h4 class="card-title">${category} - Budget Analysis</h4>
                        <div class="budget-stats">
                            <p><strong>Current Spending:</strong> ₹${adviceData.current_spent.toLocaleString('en-IN', {maximumFractionDigits: 2})}</p>
                            <p><strong>Monthly Threshold:</strong> ₹${adviceData.threshold.toLocaleString('en-IN', {maximumFractionDigits: 2})}</p>
                            ${adviceData.historical_average ? `<p><strong>Historical Average:</strong> ₹${adviceData.historical_average.toLocaleString('en-IN', {maximumFractionDigits: 2})}</p>` : ''}
                            
                            <div class="mt-3">
                                <label class="form-label">Budget Usage: ${percentageUsed}%</label>
                                <div class="progress">
                                    <div class="progress-bar ${getProgressBarClass(adviceData.status)}" 
                                         role="progressbar" 
                                         style="width: ${Math.min(percentageUsed, 100)}%"
                                         aria-valuenow="${percentageUsed}" 
                                         aria-valuemin="0" 
                                         aria-valuemax="100">
                                         ${percentageUsed}%
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mt-2">
                                <small class="text-muted">
                                    Warning Level: ${warningThreshold}% | Critical Level: ${criticalThreshold}%
                                </small>
                            </div>
                        </div>
                        
                        <div class="alert alert-${getAlertClass(adviceData.status)} mt-3">
                            ${adviceData.message || 'No specific advice available for this category.'}
                        </div>
                        ${adviceData.trend ? `<div class="alert alert-info mt-2">${adviceData.trend}</div>` : ''}
                        ${adviceData.advice && adviceData.advice.length > 0 ? `
                            <div class="mt-3">
                                <h5>Recommendations:</h5>
                                <ul class="list-unstyled">
                                    ${adviceData.advice.map(tip => `<li>• ${tip}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
            categoryAdvice.style.display = 'block';
        }

        // Load filtered transactions
        if (mainTransactionsSection) {
            mainTransactionsSection.style.display = 'block';
        }
        await loadTransactions();

    } catch (error) {
        console.error('Error:', error);
        if (categoryAdvice) {
            categoryAdvice.innerHTML = '<div class="alert alert-danger">Failed to load category advice. Please try again.</div>';
        }
    }
}

// Helper function to get progress bar class based on status
function getProgressBarClass(status) {
    switch (status) {
        case 'critical':
            return 'bg-danger';
        case 'warning':
            return 'bg-warning';
        case 'notice':
            return 'bg-info';
        default:
            return 'bg-success';
    }
}

// Helper function to get alert class based on status
function getAlertClass(status) {
    switch (status) {
        case 'critical':
            return 'danger';
        case 'warning':
            return 'warning';
        case 'notice':
            return 'info';
        default:
            return 'success';
    }
}

// Add event listener for category changes
document.addEventListener('DOMContentLoaded', function() {
    const filterCategory = document.getElementById('filterCategory');
    if (filterCategory) {
        // Ensure the dropdown is always interactive
        filterCategory.style.pointerEvents = 'auto';
        filterCategory.addEventListener('change', function(e) {
            handleCategoryChange(e);
        });
    }
});
