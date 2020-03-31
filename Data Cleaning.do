global PATH H:/downloads/ECON21130Project
global DATA $PATH/Data

*format the data tags
{
{
import delimited using $DATA/Labels/label_agegrp.csv, clear
drop if v2=="label"
rename (v1 v2) (agegrp AgeGroup)
save $DATA/CleanedLabels/age.dta, replace
}
{
import delimited using $DATA/Labels/label_education.csv, clear
drop if v2=="label"
rename (v1 v2) (education EducDescript)
save $DATA/CleanedLabels/education.dta, replace
}
{
import delimited using $DATA/Labels/label_firmage.csv, clear
drop if v2=="label"
rename (v1 v2) (firmage FirmAgeDescript)
save $DATA/CleanedLabels/firmage.dta, replace
}
{
import delimited using $DATA/Labels/label_firmsize.csv, clear
drop if v2=="label"
rename (v1 v2) (firmsize EmployeeNumber)
save $DATA/CleanedLabels/firmsize.dta, replace
}
{
import delimited using $DATA/Labels/label_industry.csv, clear
drop v3
drop if v2=="label"
rename (v1 v2) (industry IndustryDescript)
save $DATA/CleanedLabels/industry.dta, replace
}
}

import delimited $DATA\j2j_us_all.csv, clear

drop nehiresearn_dest ensepsearn_orig jobstaysearn_orig jobstaysearn_dest

drop if year < 2008

keep if periodicity == "Q"

keep if seasonadj == "S"
*can change below if we want to do state-level data
keep if geo_level == "N"

drop periodicity seasonadj geo_level


*merge on the tags
merge m:1 agegrp using $DATA/CleanedLabels/age.dta, nogen
merge m:1 education using $DATA/CleanedLabels/education.dta, keep(1 3) nogen
merge m:1 firmage using $DATA/CleanedLabels/firmage.dta, keep(1 3) nogen
merge m:1 firmsize using $DATA/CleanedLabels/firmsize.dta, keep(1 3) nogen
merge m:1 industry using $DATA/CleanedLabels/industry.dta, keep(1 3) nogen

replace IndustryDescript = "All Industries" if missing(IndustryDescript)

drop geography industry agegrp race ethnicity education education firmage firmsize

*drop the status/flag variables
drop ind_level agg_level smhire smsep smjobstart smjobend seehire seesep sj2jhire sj2jsep snehire sensep snepersist senpersist snefullq senfullq smainb smaine seeseps seehires saqseps saqhires snepersists senpersists sjobstays smainbs smaines snehiresearn_dest sensepsearn_orig sjobstaysearn_orig sjobstaysearn_dest saqhire saqsep

gen Gender = "Male" if sex == 1
replace Gender = "Female" if sex == 2
replace Gender = "All" if missing(Gender)

drop sex

gen PubPrivOwnership = "Public Only" if ownercode == "A01"
replace PubPrivOwnership = "Private Only" if ownercode == "A05"
replace PubPrivOwnership = "Mixed" if missing(PubPrivOwnership)

drop ownercode

order year quarter PubPrivOwnership AgeGroup EducDescript FirmAgeDescript EmployeeNumber IndustryDescript Gender

export excel using "$PATH/Cleaned Data.xlsx", firstrow(variables) replace
