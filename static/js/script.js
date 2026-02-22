document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', function(event) {
            let isValid = true;
            
            // Validate all required fields
            const requiredFields = form.querySelectorAll('[required]');
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('is-invalid');
                } else {
                    field.classList.remove('is-invalid');
                }
            });
            
            // Validate numerical fields
            const numericalFields = form.querySelectorAll('input[type="number"]');
            numericalFields.forEach(field => {
                if (field.value && parseFloat(field.value) < 0) {
                    isValid = false;
                    field.classList.add('is-invalid');
                }
            });
            
            if (!isValid) {
                event.preventDefault();
                alert('Please fill all required fields with valid values.');
            }
        });
    }
    
    // Dynamic form interactions
    const internetServiceSelect = document.getElementById('InternetService');
    if (internetServiceSelect) {
        internetServiceSelect.addEventListener('change', function() {
            // You can add dynamic field updates based on service selection
        });
    }
    
    // Calculate Total Charges if tenure and monthly charges are provided
    const tenureField = document.getElementById('tenure');
    const monthlyChargesField = document.getElementById('MonthlyCharges');
    const totalChargesField = document.getElementById('TotalCharges');
    
    if (tenureField && monthlyChargesField && totalChargesField) {
        function calculateTotalCharges() {
            const tenure = parseFloat(tenureField.value) || 0;
            const monthly = parseFloat(monthlyChargesField.value) || 0;
            totalChargesField.value = (tenure * monthly).toFixed(2);
        }
        
        tenureField.addEventListener('input', calculateTotalCharges);
        monthlyChargesField.addEventListener('input', calculateTotalCharges);
    }
});