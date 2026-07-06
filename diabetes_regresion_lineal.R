# ==============================================================================
# Trabajo Fin de Grado - Grado en Matemáticas
# Regresión lineal y logística: fundamentos teóricos y aplicaciones en medicina
#
# Script: Regresión lineal aplicada al conjunto de datos de diabetes
# Autor: Eduardo Calvo Alonso
# Curso: 2025-2026
#
# Descripción:
#   Este programa reproduce el análisis de regresión lineal realizado sobre el
#   conjunto de datos Diabetes de scikit-learn. Incluye análisis exploratorio,
#   ajuste de modelos lineales, diagnóstico, estudio de multicolinealidad y
#   comparación con regresión ridge y lasso.
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
  "reticulate",
  "corrplot",
  "car",
  "glmnet",
  "olsrr"
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

# El conjunto de datos se carga desde scikit-learn mediante reticulate.
# En caso de no tener scikit-learn instalado en el entorno Python asociado a R,
# puede instalarse desde R con: reticulate::py_install("scikit-learn").
reticulate::py_require(c("scikit-learn", "pandas"))
sklearn_datasets <- reticulate::import("sklearn.datasets")


# 1. Carga de datos -------------------------------------------------------------

diabetes <- sklearn_datasets$load_diabetes(as_frame = TRUE)
datos <- as_tibble(diabetes$frame)

cat("\nDimensiones del conjunto de datos:\n")
print(dim(datos))

cat("\nVariables disponibles:\n")
print(names(datos))

cat("\nResumen descriptivo inicial:\n")
print(summary(datos))


# 2. Análisis exploratorio ------------------------------------------------------

vars_predictoras <- c("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
vars_cuantitativas <- c("age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")

etiquetas_vars <- c(
  age = "Edad",
  bmi = "IMC",
  bp  = "Presión arterial media",
  s1  = "s1: colesterol total",
  s2  = "s2: LDL",
  s3  = "s3: HDL",
  s4  = "s4: colesterol total / HDL",
  s5  = "s5: log. triglicéridos",
  s6  = "s6: glucosa"
)

# 2.1. Distribución de la variable respuesta -----------------------------------

fig_target_hist <- ggplot(datos, aes(x = target)) +
  geom_histogram(bins = 25, fill = "grey80", color = "white") +
  labs(
    title = "Distribución de la variable respuesta",
    x = "Progresión de la diabetes",
    y = "Frecuencia"
  ) +
  theme_minimal()

print(fig_target_hist)

if (guardar_figuras) {
  ggsave(
    filename = file.path(ruta_figuras, "diabetes_histograma_target.png"),
    plot = fig_target_hist,
    width = 7,
    height = 5,
    dpi = 300
  )
}

# 2.2. Distribución conjunta de las variables explicativas ----------------------

datos_long <- datos %>%
  select(all_of(vars_cuantitativas)) %>%
  pivot_longer(
    cols = everything(),
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

box_stats <- datos_long %>%
  group_by(variable) %>%
  summarise(
    q1 = quantile(valor, 0.25),
    mediana = median(valor),
    q3 = quantile(valor, 0.75),
    iqr = IQR(valor),
    minimo = min(valor),
    maximo = max(valor),
    whisker_inf = max(minimo, q1 - 1.5 * iqr),
    whisker_sup = min(maximo, q3 + 1.5 * iqr),
    .groups = "drop"
  )

outliers <- datos_long %>%
  left_join(box_stats, by = "variable") %>%
  filter(valor < whisker_inf | valor > whisker_sup)

fig_distribuciones <- ggplot(datos_long, aes(x = valor)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 25,
    fill = "grey80",
    color = "white",
    linewidth = 0.25
  ) +
  geom_segment(
    data = box_stats,
    aes(x = whisker_inf, xend = whisker_sup, y = -4.3, yend = -4.3),
    inherit.aes = FALSE,
    linewidth = 0.45
  ) +
  geom_segment(
    data = box_stats,
    aes(x = whisker_inf, xend = whisker_inf, y = -5.0, yend = -3.6),
    inherit.aes = FALSE,
    linewidth = 0.45
  ) +
  geom_segment(
    data = box_stats,
    aes(x = whisker_sup, xend = whisker_sup, y = -5.0, yend = -3.6),
    inherit.aes = FALSE,
    linewidth = 0.45
  ) +
  geom_rect(
    data = box_stats,
    aes(xmin = q1, xmax = q3, ymin = -5.2, ymax = -3.4),
    inherit.aes = FALSE,
    fill = "white",
    color = "black",
    linewidth = 0.45
  ) +
  geom_segment(
    data = box_stats,
    aes(x = mediana, xend = mediana, y = -5.2, yend = -3.4),
    inherit.aes = FALSE,
    linewidth = 0.55
  ) +
  geom_point(
    data = outliers,
    aes(x = valor, y = -4.3),
    inherit.aes = FALSE,
    size = 0.5
  ) +
  facet_wrap(~ variable, ncol = 3) +
  coord_cartesian(ylim = c(-6.5, NA), clip = "off") +
  scale_y_continuous(breaks = function(x) pretty(x[x >= 0])) +
  labs(
    x = "Valor estandarizado",
    y = "Densidad"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(face = "bold", size = 10),
    axis.text.x = element_text(size = 8),
    axis.text.y = element_text(size = 8),
    axis.title.x = element_text(size = 12, margin = margin(t = 8)),
    axis.title.y = element_text(size = 12),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1.15, "lines"),
    plot.margin = margin(5.5, 10, 5.5, 10)
  )

print(fig_distribuciones)

if (guardar_figuras) {
  ggsave(
    filename = file.path(ruta_figuras, "diabetes_distribuciones_predictoras.png"),
    plot = fig_distribuciones,
    width = 9,
    height = 7,
    dpi = 300
  )
}

# 2.3. Matriz de correlaciones --------------------------------------------------

correlaciones <- cor(datos)
cat("\nMatriz de correlaciones redondeada:\n")
print(round(correlaciones, 2))

corrplot(
  correlaciones,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  number.cex = 0.85,
  tl.col = "black",
  tl.srt = 45,
  tl.cex = 1.15,
  cl.cex = 0.9,
  mar = c(0, 0, 1, 0)
)

# 2.4. Relaciones individuales con la variable respuesta -----------------------

variables_destacadas <- c("bmi", "bp", "s5")

for (var in variables_destacadas) {
  figura <- ggplot(datos, aes(x = .data[[var]], y = target)) +
    geom_point(alpha = 0.75) +
    geom_smooth(method = "lm", se = FALSE) +
    labs(
      title = paste("Progresión de la diabetes frente a", var),
      x = paste(var, "estandarizado"),
      y = "Progresión de la diabetes"
    ) +
    theme_minimal()

  print(figura)
}


# 3. Modelo lineal completo -----------------------------------------------------

modelo_completo <- lm(target ~ ., data = datos)

cat("\nResumen del modelo lineal completo:\n")
print(summary(modelo_completo))

obtener_tabla_coeficientes_lm <- function(modelo) {
  resumen <- summary(modelo)$coefficients
  ic <- confint(modelo, level = 0.95)

  data.frame(
    Variable = rownames(resumen),
    Beta = resumen[, "Estimate"],
    IC_95_inf = ic[, 1],
    IC_95_sup = ic[, 2],
    p_valor = resumen[, "Pr(>|t|)"],
    row.names = NULL
  )
}

tabla_modelo_completo <- obtener_tabla_coeficientes_lm(modelo_completo)
cat("\nCoeficientes del modelo completo:\n")
print(tabla_modelo_completo)

# Diagnóstico gráfico básico del modelo completo.
par(mfrow = c(2, 2))
plot(modelo_completo)
par(mfrow = c(1, 1))

# Multicolinealidad.
vif_completo <- car::vif(modelo_completo)
cat("\nVIF del modelo completo:\n")
print(vif_completo)


# 4. Modelo lineal reducido -----------------------------------------------------

modelo_reducido <- lm(
  target ~ age + sex + bmi + bp + s5 + s6,
  data = datos
)

cat("\nResumen del modelo lineal reducido:\n")
print(summary(modelo_reducido))

tabla_modelo_reducido <- obtener_tabla_coeficientes_lm(modelo_reducido)
cat("\nCoeficientes del modelo reducido:\n")
print(tabla_modelo_reducido)

vif_reducido <- car::vif(modelo_reducido)
cat("\nVIF del modelo reducido:\n")
print(vif_reducido)

comparar_modelos_lm <- function(modelo, nombre) {
  resumen <- summary(modelo)

  data.frame(
    Modelo = nombre,
    R2 = resumen$r.squared,
    R2_ajustado = resumen$adj.r.squared,
    AIC = AIC(modelo),
    BIC = BIC(modelo),
    row.names = NULL
  )
}

comparativa_lm <- bind_rows(
  comparar_modelos_lm(modelo_completo, "Lineal completo"),
  comparar_modelos_lm(modelo_reducido, "Lineal reducido")
)

cat("\nComparativa de modelos lineales:\n")
print(comparativa_lm)

cat("\nComparación ANOVA entre modelo reducido y completo:\n")
print(anova(modelo_reducido, modelo_completo))

cat("\nCp de Mallows del modelo completo:\n")
print(olsrr::ols_mallows_cp(modelo_completo, modelo_completo))

cat("\nCp de Mallows del modelo reducido:\n")
print(olsrr::ols_mallows_cp(modelo_reducido, modelo_completo))


# 5. Regresión ridge y lasso ----------------------------------------------------

x <- as.matrix(datos[, vars_predictoras])
y <- datos$target

# 5.1. Ridge --------------------------------------------------------------------

cv_ridge <- cv.glmnet(
  x = x,
  y = y,
  alpha = 0,
  nfolds = 10
)

cat("\nLambda seleccionado para ridge:\n")
print(cv_ridge$lambda.min)

cat("\nCoeficientes ridge para lambda.min:\n")
print(coef(cv_ridge, s = "lambda.min"))

plot(cv_ridge)

# 5.2. Lasso --------------------------------------------------------------------

cv_lasso <- cv.glmnet(
  x = x,
  y = y,
  alpha = 1,
  nfolds = 10
)

cat("\nLambda seleccionado para lasso:\n")
print(cv_lasso$lambda.min)

cat("\nCoeficientes lasso para lambda.min:\n")
print(coef(cv_lasso, s = "lambda.min"))

cat("\nCoeficientes lasso para lambda.1se:\n")
print(coef(cv_lasso, s = "lambda.1se"))

plot(cv_lasso)


# 6. Evaluación predictiva mediante partición entrenamiento/test ---------------

train_index <- sample(seq_len(nrow(datos)), size = floor(0.8 * nrow(datos)))
train <- datos[train_index, ]
test <- datos[-train_index, ]

mse <- function(real, predicho) {
  predicho <- as.numeric(predicho)
  mean((real - predicho)^2)
}

# 6.1. Modelo lineal completo.
modelo_completo_train <- lm(target ~ ., data = train)
pred_lm <- predict(modelo_completo_train, newdata = test)
mse_lm <- mse(test$target, pred_lm)

# 6.2. Modelo lineal reducido.
modelo_reducido_train <- lm(
  target ~ age + sex + bmi + bp + s5 + s6,
  data = train
)
pred_reducido <- predict(modelo_reducido_train, newdata = test)
mse_reducido <- mse(test$target, pred_reducido)

# 6.3. Ridge.
x_train <- as.matrix(train[, vars_predictoras])
y_train <- train$target
x_test <- as.matrix(test[, vars_predictoras])
y_test <- test$target

cv_ridge_train <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 0,
  nfolds = 10
)

pred_ridge <- predict(
  cv_ridge_train,
  s = "lambda.min",
  newx = x_test
)
mse_ridge <- mse(y_test, pred_ridge)

# 6.4. Lasso.
cv_lasso_train <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 10
)

pred_lasso <- predict(
  cv_lasso_train,
  s = "lambda.min",
  newx = x_test
)
mse_lasso <- mse(y_test, pred_lasso)

comparativa_mse <- data.frame(
  Modelo = c(
    "Lineal completo",
    "Lineal reducido",
    "Ridge",
    "Lasso"
  ),
  MSE = c(
    mse_lm,
    mse_reducido,
    mse_ridge,
    mse_lasso
  )
)

cat("\nComparativa predictiva final en el conjunto de test:\n")
print(comparativa_mse)

# Fin del script ----------------------------------------------------------------
