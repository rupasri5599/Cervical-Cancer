function predictBiopsy() {
    let age = parseFloat(document.getElementById("age").value);
    let partners = parseFloat(document.getElementById("partners").value);
    let pregnancies = parseFloat(document.getElementById("pregnancies").value);
    let smokes = parseFloat(document.getElementById("smokes").value);
    let contraceptives = parseFloat(document.getElementById("contraceptives").value);

    // Standardization (using the same scaler values from training)
    let meanValues = [32.47, 2.52, 2.18, 0.15, 0.34];
    let stdValues = [11.55, 1.66, 1.64, 0.36, 0.47];
    
    let standardizedInput = [
        (age - meanValues[0]) / stdValues[0],
        (partners - meanValues[1]) / stdValues[1],
        (pregnancies - meanValues[2]) / stdValues[2],
        (smokes - meanValues[3]) / stdValues[3],
        (contraceptives - meanValues[4]) / stdValues[4]
    ];

    // Model Coefficients (from trained model)
    let coefficients = [-0.21, 0.13, 0.07, 0.46, -0.12];
    let intercept = -2.19;

    // Calculate probability (logistic regression function)
    let linearSum = intercept;
    for (let i = 0; i < coefficients.length; i++) {
        linearSum += standardizedInput[i] * coefficients[i];
    }
    let probability = 1 / (1 + Math.exp(-linearSum));
    let prediction = probability >= 0.5 ? 1 : 0;

    // Display result
    document.getElementById("result").innerText = prediction;
}