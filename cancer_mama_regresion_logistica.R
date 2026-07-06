# ==============================================================================
# Trabajo Fin de Grado - Grado en Matemáticas
# Regresión lineal y logística: fundamentos teóricos y aplicaciones en medicina
#
# Script: Regresión logística aplicada al cáncer de mama
# Autor: Eduardo Calvo Alonso
# Curso: 2025-2026
#
# Descripción:
#   Este programa reproduce el análisis de regresión logística realizado sobre el
#   conjunto Breast Cancer Wisconsin. Incluye análisis exploratorio, ajuste de un
#   modelo logístico completo y reducido, evaluación mediante ROC/AUC, validación
#   cruzada y comparación con ridge y lasso.
# ==============================================================================

# 0. Configuración inicial -------------------------------------------------------

set.seed(123)

# Cambiar a TRUE si se desea guardar algunas figuras en la carpeta figures/.
guardar_figuras <- FALSE
ruta_figuras <- "figures"

if (guardar_figuras && !dir.exists(ruta_figuras)) {
  dir.create(ruta_figuras, recursive = TRUE)
}

paquetes <- c(
  "tidyverse",
  "mlbench",
  "corrplot",
  "pROC",
  "caret",
  "pscl",
  "glmnet"
)

invisible(lapply(paquetes, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      paste0(
        "Falta el paquete '", pkg,
        "'. Instálalo con install.packages('", pkg, "')."
      ),
      call. = FALSE
    )
  }
  library(pkg, character.only = TRUE)
}))


# 1. Carga y preparación de datos ------------------------------------------------

data("BreastCancer", package = "mlbench")
datos <- BreastCancer

# Eliminamos el identificador, que no aporta información predictiva.
datos <- datos %>%
  select(-Id) %>%
  na.omit()

# Conversión de variables explicativas a formato numérico.
datos <- datos %>%
  mutate(across(.cols = -Class, .fns = ~ as.numeric(as.character(.))))

# Nombres en español.
colnames(datos) <- c(
  "Espesor_celular",
  "Tamano_celular",
  "Forma_celular",
  "Adhesion_marginal",
  "Tam_epitelial",
  "Nucleos_desnudos",
  "Cromatina",
  "Nucleolos",
  "Mitosis",
  "Diagnostico"
)

# Variable respuesta en formato factor y binario.
datos <- datos %>%
  mutate(
    Diagnostico = ifelse(Diagnostico == "malignant", "Maligno", "Benigno"),
    Diagnostico = factor(Diagnostico, levels = c("Benigno", "Maligno")),
    Diagnostico_bin = ifelse(Diagnostico == "Maligno", 1, 0)
  )

cat("\nDimensiones del conjunto de datos tras eliminar valores perdidos:\n")
print(dim(datos))

cat("\nDistribución de clases:\n")
print(table(datos$Diagnostico))
print(prop.table(table(datos$Diagnostico)))


# 2. Análisis exploratorio ------------------------------------------------------

vars_cancer <- c(
  "Espesor_celular",
  "Tamano_celular",
  "Forma_celular",
  "Adhesion_marginal",
  "Tam_epitelial",
  "Nucleos_desnudos",
  "Cromatina",
  "Nucleolos",
  "Mitosis"
)

etiquetas_vars <- c(
  Espesor_celular = "Espesor celular",
  Tamano_celular = "Tamaño celular",
  Forma_celular = "Forma celular",
  Adhesion_marginal = "Adhesión marginal",
  Tam_epitelial = "Tamaño epitelial",
  Nucleos_desnudos = "Núcleos desnudos",
  Cromatina = "Cromatina",
  Nucleolos = "Nucléolos",
  Mitosis = "Mitosis"
)

# 2.1. Distribución de clases ---------------------------------------------------

fig_clases <- ggplot(datos, aes(x = Diagnostico, fill = Diagnostico)) +
  geom_bar() +
  labs(
    title = "Distribución de tumores",
    x = "Diagnóstico",
    y = "Frecuencia"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

print(fig_clases)

if (guardar_figuras) {
  ggsave(
    filename = file.path(ruta_figuras, "cancer_distribucion_clases.png"),
    plot = fig_clases,
    width = 6,
    height = 4,
    dpi = 300
  )
}

# 2.2. Boxplots conjuntos por diagnóstico --------------------------------------

datos_long_cancer <- datos %>%
  select(Diagnostico, all_of(vars_cancer)) %>%
  pivot_longer(
    cols = all_of(vars_cancer),
    names_to = "variable",
    values_to = "valor"
  ) %>%
  mutate(
    variable = factor(
      variable,
      levels = names(etiquetas_vars),
      labels = unname(etiquetas_vars)
    )
  )

fig_distribucion_cancer <- ggplot(
  datos_long_cancer,
  aes(x = valor, y = Diagnostico, fill = Diagnostico)
) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA,
    alpha = 0.75
  ) +
  geom_jitter(
    aes(color = Diagnostico),
    height = 0.13,
    width = 0.08,
    alpha = 0.18,
    size = 0.45,
    show.legend = FALSE
  ) +
  facet_wrap(~ variable, ncol = 3) +
  scale_x_continuous(breaks = 1:10, limits = c(1, 10)) +
  labs(
    x = "Valor de la variable",
    y = "Diagnóstico"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 9),
    axis.title.x = element_text(size = 12, margin = margin(t = 8)),
    axis.title.y = element_text(size = 12),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1.15, "lines"),
    legend.position = "none"
  )

print(fig_distribucion_cancer)

if (guardar_figuras) {
  ggsave(
    filename = file.path(ruta_figuras, "cancer_boxplots_variables.png"),
    plot = fig_distribucion_cancer,
    width = 9,
    height = 7,
    dpi = 300
  )
}

# 2.3. Matriz de correlaciones --------------------------------------------------

correlaciones <- datos %>%
  select(all_of(vars_cancer)) %>%
  cor()

cat("\nMatriz de correlaciones redondeada:\n")
print(round(correlaciones, 2))

corrplot(
  correlaciones,
  method = "color",
  addCoef.col = "black",
  type = "upper",
  tl.srt = 45,
  tl.cex = 0.9,
  cl.cex = 1.1
)


# 3. Modelo logístico completo --------------------------------------------------

formula_completa <- as.formula(
  paste("Diagnostico_bin ~", paste(vars_cancer, collapse = " + "))
)

modelo_logistico <- glm(
  formula_completa,
  data = datos,
  family = "binomial"
)

cat("\nResumen del modelo logístico completo:\n")
print(summary(modelo_logistico))

obtener_tabla_or <- function(modelo) {
  resumen <- summary(modelo)$coefficients
  ic <- suppressMessages(confint(modelo))

  data.frame(
    Variable = rownames(resumen),
    OR = exp(coef(modelo)),
    IC_95_inf = exp(ic[, 1]),
    IC_95_sup = exp(ic[, 2]),
    p_valor = resumen[, "Pr(>|z|)"],
    row.names = NULL
  )
}

tabla_or_completo <- obtener_tabla_or(modelo_logistico)
cat("\nOdds ratio del modelo completo:\n")
print(tabla_or_completo)


# 4. Funciones de evaluación ----------------------------------------------------

calcular_metricas <- function(real, prob, umbral = 0.5) {
  prob <- as.numeric(prob)
  pred <- ifelse(prob > umbral, "Maligno", "Benigno")
  pred <- factor(pred, levels = c("Benigno", "Maligno"))

  tabla <- table(Real = real, Predicho = pred)

  accuracy <- sum(diag(tabla)) / sum(tabla)
  sensibilidad <- tabla["Maligno", "Maligno"] / sum(tabla["Maligno", ])
  especificidad <- tabla["Benigno", "Benigno"] / sum(tabla["Benigno", ])

  auc_valor <- as.numeric(
    pROC::auc(
      pROC::roc(
        response = real,
        predictor = as.numeric(prob),
        levels = c("Benigno", "Maligno"),
        quiet = TRUE
      )
    )
  )

  data.frame(
    Accuracy = accuracy,
    Sensibilidad = sensibilidad,
    Especificidad = especificidad,
    AUC = auc_valor
  )
}

prob_completo <- predict(modelo_logistico, type = "response")
metricas_completo <- calcular_metricas(datos$Diagnostico, prob_completo)

cat("\nPseudo-R2 del modelo completo:\n")
print(pscl::pR2(modelo_logistico))

cat("\nMétricas del modelo completo evaluado sobre la muestra completa:\n")
print(metricas_completo)

roc_completo <- pROC::roc(
  response = datos$Diagnostico,
  predictor = prob_completo,
  levels = c("Benigno", "Maligno"),
  quiet = TRUE
)

plot(
  roc_completo,
  main = "Curva ROC - Modelo logístico completo",
  legacy.axes = TRUE
)


# 5. Modelo reducido ------------------------------------------------------------

# Selección automática por AIC mediante stepwise bidireccional.
modelo_reducido <- step(
  modelo_logistico,
  direction = "both",
  trace = FALSE
)

cat("\nResumen del modelo logístico reducido:\n")
print(summary(modelo_reducido))

cat("\nComparación AIC:\n")
print(AIC(modelo_logistico, modelo_reducido))

tabla_or_reducido <- obtener_tabla_or(modelo_reducido)
cat("\nOdds ratio del modelo reducido:\n")
print(tabla_or_reducido)

prob_reducido <- predict(modelo_reducido, type = "response")
metricas_reducido <- calcular_metricas(datos$Diagnostico, prob_reducido)

cat("\nPseudo-R2 del modelo reducido:\n")
print(pscl::pR2(modelo_reducido))

cat("\nMétricas del modelo reducido evaluado sobre la muestra completa:\n")
print(metricas_reducido)

roc_reducido <- pROC::roc(
  response = datos$Diagnostico,
  predictor = prob_reducido,
  levels = c("Benigno", "Maligno"),
  quiet = TRUE
)

plot(
  roc_completo,
  main = "Comparación ROC: modelo completo y reducido",
  legacy.axes = TRUE
)
plot(roc_reducido, add = TRUE, lwd = 2)
legend(
  "bottomright",
  legend = c(
    paste0("Completo, AUC = ", round(pROC::auc(roc_completo), 3)),
    paste0("Reducido, AUC = ", round(pROC::auc(roc_reducido), 3))
  ),
  lwd = 2,
  bty = "n"
)


# 6. Regularización: ridge y lasso ---------------------------------------------

x <- model.matrix(formula_completa, data = datos)[, -1]
y <- datos$Diagnostico_bin

# 6.1. Ridge --------------------------------------------------------------------

set.seed(123)

cv_ridge <- cv.glmnet(
  x = x,
  y = y,
  family = "binomial",
  alpha = 0,
  nfolds = 10,
  type.measure = "auc"
)

cat("\nLambda seleccionado para ridge:\n")
print(cv_ridge$lambda.min)

cat("\nCoeficientes ridge para lambda.min:\n")
print(coef(cv_ridge, s = "lambda.min"))

prob_ridge <- predict(
  cv_ridge,
  newx = x,
  s = "lambda.min",
  type = "response"
)

metricas_ridge <- calcular_metricas(datos$Diagnostico, prob_ridge)
cat("\nMétricas ridge evaluadas sobre la muestra completa:\n")
print(metricas_ridge)

# 6.2. Lasso --------------------------------------------------------------------

set.seed(123)

cv_lasso <- cv.glmnet(
  x = x,
  y = y,
  family = "binomial",
  alpha = 1,
  nfolds = 10,
  type.measure = "auc"
)

cat("\nLambda seleccionado para lasso:\n")
print(cv_lasso$lambda.min)

cat("\nCoeficientes lasso para lambda.min:\n")
print(coef(cv_lasso, s = "lambda.min"))

cat("\nCoeficientes lasso para lambda.1se:\n")
print(coef(cv_lasso, s = "lambda.1se"))

prob_lasso <- predict(
  cv_lasso,
  newx = x,
  s = "lambda.1se",
  type = "response"
)

metricas_lasso <- calcular_metricas(datos$Diagnostico, prob_lasso)
cat("\nMétricas lasso evaluadas sobre la muestra completa:\n")
print(metricas_lasso)

coef_lasso_1se <- as.matrix(coef(cv_lasso, s = "lambda.1se"))
variables_lasso <- rownames(coef_lasso_1se)[coef_lasso_1se[, 1] != 0]
variables_lasso <- setdiff(variables_lasso, "(Intercept)")

cat("\nVariables no nulas retenidas por lasso con lambda.1se:\n")
print(variables_lasso)


# 7. Validación cruzada común ---------------------------------------------------

# Se emplea una validación cruzada externa estratificada de 10 particiones.
# En ridge y lasso, lambda se selecciona de nuevo dentro de cada conjunto de
# entrenamiento mediante cv.glmnet, evitando usar información del subconjunto de
# validación externo durante la elección del parámetro.

set.seed(123)

folds <- caret::createFolds(
  datos$Diagnostico,
  k = 10,
  returnTrain = FALSE
)

resultados_cv <- data.frame()

for (j in seq_along(folds)) {
  test_idx <- folds[[j]]

  train <- datos[-test_idx, ]
  test <- datos[test_idx, ]

  x_train <- model.matrix(formula_completa, data = train)[, -1]
  x_test <- model.matrix(formula_completa, data = test)[, -1]
  y_train <- train$Diagnostico_bin

  # 7.1. Modelo logístico clásico.
  modelo_glm <- glm(
    formula_completa,
    data = train,
    family = "binomial"
  )

  prob_glm <- predict(
    modelo_glm,
    newdata = test,
    type = "response"
  )

  met_glm <- calcular_metricas(test$Diagnostico, prob_glm)
  met_glm$Modelo <- "Logístico clásico"
  met_glm$Fold <- j

  # 7.2. Ridge.
  cv_ridge_fold <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = 0,
    nfolds = 10,
    type.measure = "auc"
  )

  prob_ridge_fold <- predict(
    cv_ridge_fold,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )

  met_ridge <- calcular_metricas(test$Diagnostico, prob_ridge_fold)
  met_ridge$Modelo <- "Ridge"
  met_ridge$Fold <- j

  # 7.3. Lasso.
  cv_lasso_fold <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = 1,
    nfolds = 10,
    type.measure = "auc"
  )

  prob_lasso_fold <- predict(
    cv_lasso_fold,
    newx = x_test,
    s = "lambda.1se",
    type = "response"
  )

  met_lasso <- calcular_metricas(test$Diagnostico, prob_lasso_fold)
  met_lasso$Modelo <- "Lasso"
  met_lasso$Fold <- j

  resultados_cv <- bind_rows(
    resultados_cv,
    met_glm,
    met_ridge,
    met_lasso
  )
}

resumen_cv <- resultados_cv %>%
  group_by(Modelo) %>%
  summarise(
    Accuracy_media = mean(Accuracy),
    Sensibilidad_media = mean(Sensibilidad),
    Especificidad_media = mean(Especificidad),
    AUC_medio = mean(AUC),
    .groups = "drop"
  )

resumen_cv_sd <- resultados_cv %>%
  group_by(Modelo) %>%
  summarise(
    Accuracy_media = mean(Accuracy),
    Accuracy_sd = sd(Accuracy),
    Sensibilidad_media = mean(Sensibilidad),
    Sensibilidad_sd = sd(Sensibilidad),
    Especificidad_media = mean(Especificidad),
    Especificidad_sd = sd(Especificidad),
    AUC_medio = mean(AUC),
    AUC_sd = sd(AUC),
    .groups = "drop"
  )

cat("\nResumen de validación cruzada:\n")
print(resumen_cv)

cat("\nResumen de validación cruzada con desviaciones típicas:\n")
print(resumen_cv_sd)

# Fin del script ----------------------------------------------------------------
