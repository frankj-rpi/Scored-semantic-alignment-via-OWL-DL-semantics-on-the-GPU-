import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.IRI;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLNamedIndividual;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.InferenceType;
import org.semanticweb.owlapi.reasoner.OWLReasoner;

public final class ElkOracleRunner {
    private ElkOracleRunner() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && "--probe".equals(args[0])) {
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
            manager.getOWLDataFactory();
            new ElkReasonerFactory();
            System.out.println("PROBE_OK");
            return;
        }

        if (args.length != 3) {
            System.err.println("Usage: ElkOracleRunner <ontology.owl> <targets.tsv> <candidates.txt>");
            System.exit(2);
        }

        Path ontologyPath = Path.of(args[0]);
        Path targetsPath = Path.of(args[1]);
        Path candidatesPath = Path.of(args[2]);

        Map<String, String> targetToQueryClass = new LinkedHashMap<>();
        try (BufferedReader reader = Files.newBufferedReader(targetsPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split("\t", 2);
                if (parts.length != 2) {
                    throw new IllegalArgumentException("Invalid target mapping line: " + line);
                }
                targetToQueryClass.put(parts[0], parts[1]);
            }
        }

        Set<String> candidates = new LinkedHashSet<>();
        try (BufferedReader reader = Files.newBufferedReader(candidatesPath, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.isEmpty()) {
                    candidates.add(line);
                }
            }
        }

        long setupStartNanos = System.nanoTime();
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(new File(ontologyPath.toString()));
        OWLReasoner reasoner = new ElkReasonerFactory().createReasoner(ontology);
        long setupElapsedMillis = Math.round((System.nanoTime() - setupStartNanos) / 1_000_000.0);

        long startNanos = System.nanoTime();
        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY, InferenceType.CLASS_ASSERTIONS);
        long precomputeElapsedMillis = Math.round((System.nanoTime() - startNanos) / 1_000_000.0);

        long queryStartNanos = System.nanoTime();
        for (Map.Entry<String, String> entry : targetToQueryClass.entrySet()) {
            String targetIri = entry.getKey();
            String queryClassIri = entry.getValue();
            OWLClass queryClass = manager.getOWLDataFactory().getOWLClass(IRI.create(queryClassIri));
            for (OWLNamedIndividual individual : reasoner.getInstances(queryClass, false).getFlattened()) {
                String iri = individual.getIRI().toString();
                if (candidates.contains(iri)) {
                    System.out.println("MEMBER\t" + targetIri + "\t" + iri);
                }
            }
        }
        long queryElapsedMillis = Math.round((System.nanoTime() - queryStartNanos) / 1_000_000.0);

        System.out.println("ELAPSED_MS\t" + (setupElapsedMillis + precomputeElapsedMillis + queryElapsedMillis));
        System.out.println("SETUP_MS\t" + setupElapsedMillis);
        System.out.println("PREPROCESS_MS\t" + precomputeElapsedMillis);
        System.out.println("POSTPROCESS_MS\t" + queryElapsedMillis);

        reasoner.dispose();
    }
}
