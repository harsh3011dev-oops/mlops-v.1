{{/*
Expand the name of the chart.
*/}}
{{- define "houseprice-estimator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "houseprice-estimator.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "houseprice-estimator.labels" -}}
helm.sh/chart: {{ include "houseprice-estimator.name" . }}-{{ .Chart.Version | replace "+" "_" }}
{{ include "houseprice-estimator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "houseprice-estimator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "houseprice-estimator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
