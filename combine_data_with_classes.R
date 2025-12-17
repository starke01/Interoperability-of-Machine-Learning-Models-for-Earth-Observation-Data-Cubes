library(sf)
library(dplyr)
library(readr)
library(stringr)



csv <- read_csv2("..data_frace_labels_csv..")
class9 <- readr::read_csv("https://breizhcrops.s3.eu-central-1.amazonaws.com/classmapping.csv")

data_geojson <- st_read("..area..")
tmp <- as.character(data_geojson$CODE_CULTU)
data_geojson$CODE_CULTU <- stringr::str_trim(tmp)

table(data_geojson)

data_code_cultu <- csv$`CODE_CULTU`
chr_cultu <- as.character(data_code_cultu)
code <- stringr::str_trim(chr_cultu)


label_fr    <- csv$`Libellé Culture`
group_id    <- csv$`Code Groupe Culture`
group_label <- csv$`Libellé Groupe Culture`



codes <- data.frame(
  code = code,
  label_fr = label_fr,
  group_id = group_id,
  group_label = group_label,
  stringsAsFactors = FALSE
)

class9$code <- stringr::str_trim(class9$code)



length(class9)

geo_9 <- dplyr::inner_join(data_geojson, class9, by = c("CODE_CULTU" = "code"))

geo_9 <- dplyr::left_join(geo_9, codes, by = c("CODE_CULTU" = "code"))



geo_9 <- dplyr::rename(geo_9, class_id = id,class_name = classname)


drop_geom <- st_drop_geometry(data_geojson)
geo_dropped <- dplyr::anti_join(drop_geom, class9, by = c("CODE_CULTU" = "code"))

nrow(geo_dropped)
head(geo_dropped)


drop_geom9 <- st_drop_geometry(geo_9)
class_counts <- dplyr::count(drop_geom9, class_id, class_name, sort = TRUE)
class_counts

#path drop


is_path <- names(geo_9) %in% "path"
cols_to_keep <- !is_path
geo_9 <- geo_9[, cols_to_keep]


st_write(geo_9, "..path..", driver = "GeoJSON", delete_dsn = TRUE)

table(geo_9$class_name)


s <- read_sf("data_path")
st_crs(s)
s_4326 <- st_transform(s, 4326)

st_write(
  s_4326, "..path..", driver = "GeoJSON", delete_dsn = TRUE)


