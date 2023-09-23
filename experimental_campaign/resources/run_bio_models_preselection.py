import csv
import importlib
import jax
import jax.numpy as jnp
import libsbml
import os
import sbmltoodejax
import taxoniq
import time

jax.config.update("jax_platform_name", "cpu")
def get_taxon_ids(element):
    ids = []
    num_terms = element.getNumCVTerms()
    for i in range(num_terms):
        term = element.getCVTerm(i)
        if term.getQualifierType() == libsbml.BIOLOGICAL_QUALIFIER and libsbml.BiolQualifierType_toString(term.getBiologicalQualifierType()) == "hasTaxon":
            num_resources = term.getNumResources()
            for j in range(num_resources):
                resource = term.getResourceURI(j)
                if "http://identifiers.org/taxonomy" in resource:
                    ids.append(int(resource.split("/")[-1]))
    return ids

def load_sbml_model(model_idx):

    if model_idx in [649, 694, 992, 993]:
        return False, Exception(f"Model #{model_idx} does not exits!")

    else:
        sbml_model_filepath = f"sbml_files/BIOMD{model_idx:010d}.xml"
        if not os.path.exists(sbml_model_filepath):
            model_xml_body = sbmltoodejax.biomodels_api.get_content_for_model(model_idx)
            with open(sbml_model_filepath, 'w') as sbml_model_file:
                sbml_model_file.write(model_xml_body)
        return True, None

def load_jax_model(model_idx, deltaT, atol, rtol, mxstep):
    sbml_model_filepath = f"sbml_files/BIOMD{model_idx:010d}.xml"
    jax_model_filepath = f"jax_files/BIOMD{model_idx:010d}.py"

    try:
        model_data = sbmltoodejax.parse.ParseSBMLFile(sbml_model_filepath)
    except Exception as e:
        return False, e

    try:
        sbmltoodejax.modulegeneration.GenerateModel(
            model_data,
            jax_model_filepath,
            deltaT=deltaT,
            atol=atol,
            rtol=rtol,
            mxstep=mxstep,
        )
    except Exception as e:
        os.remove(jax_model_filepath)
        return False, e

    try:
        spec = importlib.util.spec_from_file_location("JaxModelSpec", jax_model_filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_cls = getattr(module, "ModelRollout")
        model = model_cls(atol=atol, rtol=rtol, mxstep=mxstep)
        y0 = getattr(module, "y0")
        w0 = getattr(module, "w0")
        c = getattr(module, "c")
    except Exception as e:
        return False, e

    return True, (model, y0, w0, c)



if __name__ == '__main__':
    jax.config.update("jax_platform_name", "cpu")

    deltaT = 0.1
    atol = 1e-6
    rtol = 1e-12
    mxstep = 1000
    n_secs = 10
    n_system_steps = int(n_secs / deltaT)

    os.makedirs("sbml_files", exist_ok=True)
    os.makedirs("jax_files", exist_ok=True)


    model_stats_filepath = "bio_models_preselection_statistics.csv"
    if os.path.exists(model_stats_filepath):
        raise FileExistsError

    fieldnames = ['model_idx', 'loading_sbml_error', 'loading_jax_error', 'init_error', 'simu_error',
                  'model_name',
                  'species_names', 'genus_names', 'family_names', 'order_names', 'class_names', 'phylum_names', 'kingdom_names', 'superkingdom_names',
                  'n_species', 'n_parameters', 'n_compartments', 'n_reactions', 'n_raterules', 'n_assignmentrules',
                  'n_nodes', 'n_inputs_per_node', 'n_variables', 'n_constants', f"simu_time (T={n_secs})"]

    with open(model_stats_filepath, 'w') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()

    for model_idx in range(1, 1049):
        print(model_idx)
        row = {k: [] if "_names" in k else "" for k in fieldnames}
        row["model_idx"] = model_idx

        loading_sbml_success, loading_sbml_error = load_sbml_model(model_idx)
        if not loading_sbml_success:
            row["loading_sbml_error"] = str(loading_sbml_error)
            if row["loading_sbml_error"] == "":
                row["loading_sbml_error"] = "Empty Error Message"

        # Load Model
        loading_jax_success, loading_jax_results = load_jax_model(model_idx, deltaT, atol, rtol, mxstep)

        if loading_jax_success:
            model, y0, w0, c = loading_jax_results

            ## Save some statistics ---------------------------------------
            ### Parse SBML File with libSBML
            sbml_model_filepath = f"./sbml_files/BIOMD{model_idx:010d}.xml"
            doc = libsbml.readSBML(sbml_model_filepath)
            libsbml_model = doc.getModel()
            row["model_name"] = libsbml_model.getName()

            ### Get taxonomy info from the NCBI databse
            taxon_ids = get_taxon_ids(libsbml_model)
            for taxon_idx in taxon_ids:
                t = taxoniq.Taxon(taxon_idx)
                if len(t.ranked_lineage) > 0:
                    for t in t.ranked_lineage:
                        row[f"{t.rank.name}_names"].append(t.scientific_name)

            ### Get sbml info
            row["n_species"] = libsbml_model.getNumSpecies()
            row["n_parameters"] = libsbml_model.getNumParameters()
            row["n_compartments"] = libsbml_model.getNumCompartments()
            row["n_reactions"] = libsbml_model.getNumReactions()
            n_raterules = 0
            n_assignmentrules = 0
            for rule_idx in range(libsbml_model.getNumRules()):
                rule = libsbml_model.getRule(rule_idx)
                if rule.isRate():
                    n_raterules += 1
                elif rule.isAssignment():
                    n_assignmentrules += 1
            row["n_raterules"] = n_raterules
            row["n_assignmentrules"] = n_assignmentrules

            row["n_nodes"] = len(y0)
            if len(y0) > 0:
                row["n_inputs_per_node"] = jnp.abs(model.modelstepfunc.ratefunc.stoichiometricMatrix).sum(1).astype("int32").tolist()
            else:
                row["n_inputs_per_node"] = []
            row["n_variables"] = len(w0)
            row["n_constants"] = len(c)
            ## ---------------------------------------

            # Check Init
            if jnp.isnan(y0).any() or jnp.isnan(w0).any() or jnp.isnan(c).any():
                row["init_error"] = "NaN values"
            elif (y0 < 0).sum() > 0:
                row["init_error"] = "Neg values"

            # Simulate Model with default init for n_secs
            if row["init_error"] == "":
                try:
                    simu_start = time.time()
                    ys, ws, times = model(n_system_steps)
                    ys.block_until_ready()
                    simu_end = time.time()

                    if jnp.isnan(ys).any() or jnp.isnan(ws).any():
                        row["simu_error"] = "NaN values"
                    elif (ys < 0).sum() > 0:
                        row["simu_error"] = "Neg values"
                except Exception as e:
                    row["simu_error"] = str(e)
                    if row["simu_error"] == "":
                        row["simu_error"] = "Empty Error Message"

                if row["simu_error"] == "":
                    row[f"simu_time (T={n_secs})"] = simu_end - simu_start


        else:
            row["loading_jax_error"] = str(loading_jax_results)
            if row["loading_jax_error"] == "":
                row["loading_jax_error"] = "Empty Error Message"

        with open(model_stats_filepath, 'a') as f:
            writer = csv.DictWriter(f, fieldnames)
            writer.writerow(row)


