import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
	from collections import Counter
	import re
	from sentence_transformers import SentenceTransformer
	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_score
	import warnings
	import numpy as np
	from collections import defaultdict
	from sklearn.preprocessing import normalize
	from hdbscan import HDBSCAN
	from sklearn.metrics.pairwise import cosine_similarity
	import nltk
	from nltk.corpus import wordnet as wn
except:
	print("Import Error! Some packages aren't installed so import failed.")

# make sure wordnet is available
try:
	wn.ensure_loaded()
except:
	print("Error: nltk.wordnet isn't available, downloading... Please run "
	      "again.")
	nltk.download('wordnet')
	nltk.download('omw-1.4')

# don't show warnings about 'will delete this type\function in a future
# upgrade' of the library
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".force_all_finite.")

# very common meaningless words to ignore when extracting real terms from
# meanings str
_WORD_STOP = {
	'the', 'and', 'for', 'with', 'from', 'into', 'upon', 'over', 'under',
	'in',
	'on',
	'of', 'to', 'by', 'as', 'at', 'or', 'an', 'a', 'is', 'are', 'be', 'being',
	'been',
	'archaic'
}

# extremely generic wordnet heads that rarely help labeling - those are the
# highest in the hypernyms hierarchy
_TOO_GENERIC = {
	'entity', 'physical entity', 'object', 'whole', 'unit', 'organism',
	'living thing',
	'animate thing', 'animal', 'chordate', 'vertebrate', 'creation',
	'artifact', 'artefact',
	'person', 'someone', 'somebody', 'mortal', 'soul'
}

# canonical heads we try to map tokens to (pinned to intended wn synsets id)
_CANON_HEADS = {
	'bird'            : wn.synset('bird.n.01'),
	'fish'            : wn.synset('fish.n.01'),
	'mammal'          : wn.synset('mammal.n.01'),
	'insect'          : wn.synset('insect.n.01'),
	'reptile'         : wn.synset('reptile.n.01'),
	'amphibian'       : wn.synset('amphibian.n.01'),
	'plant'           : wn.synset('plant.n.02'),  # flora, not factory
	'metal'           : wn.synset('metallic_element.n.01'),
	'chemical element': wn.synset('chemical_element.n.01'),
	'clothing'        : wn.synset('clothing.n.01'),
	'food'            : wn.synset('food.n.01'),
	'motion'          : wn.synset('motion.n.01'),
	'action'          : wn.synset('action.n.01'),
	'movement'        : wn.synset('movement.n.01'),
	'bamboo'          : wn.synset('bamboo.n.01'),
	'speech'          : wn.synset('speech.n.02'),
	'weather'         : wn.synset('weather.n.01'),
	'river'           : wn.synset('river.n.01'),
	'fire'            : wn.synset('fire.n.01')
}

# heads → hypernyms for display (avoid weird wn defaults)
_CANON_HEAD_DISPLAY_HYPERNYMS = {
	'bird'            : ['vertebrate'],
	'fish'            : ['aquatic vertebrate'],
	'mammal'          : ['vertebrate'],
	'insect'          : ['arthropod'],
	'reptile'         : ['vertebrate'],
	'amphibian'       : ['vertebrate'],
	'plant'           : ['organism', 'flora'],
	'metal'           : ['chemical element'],
	'chemical element': ['substance'],
	'clothing'        : ['covering'],
	'food'            : ['substance']
}

# aliases / fine-grained categories that map to canonical heads
_ALIAS_TO_HEAD = {
	# birds
	'raptor'        : 'bird', 'vulture': 'bird',
	'condor'        : 'bird',
	'kingfisher'    : 'bird', 'pheasant': 'bird', 'snipe': 'bird',
	'passerine'     : 'bird', 'owl': 'bird', 'hawk': 'bird',
	# fish
	'eel'           : 'fish', 'anchovy': 'fish', 'shark': 'fish',
	'ray'           : 'fish',
	'saltwater fish': 'fish', 'freshwater fish': 'fish',
	'frog'          : 'amphibian',
	# mammals
	'equine'        : 'equine', 'equid': 'horse', 'horse': 'equine',
	'mule'          : 'equine',
	# verbs
	'act'           : 'action', 'move': 'movement', 'talk': 'speech',
}

# ******** HELPER FUNCS **************
def _semantic_heads_from_meanings(cluster_meanings,
                                  top_k_heads=3,
                                  min_head_share=0.10,
                                  dominance_pair_cover=0.70):
	"""
    infer compact semantic heads (like 'fish', 'bird') from meaning strings
    uses token frequency+wordnet mapping and applies a dominance heuristic
    :param cluster_meanings: list of meaning strings for a cluster
    :param top_k_heads: maximum heads to keep after filtering
    :param min_head_share: minimum vote share for a head to be kept
    :param dominance_pair_cover: if top-2 heads together ≥ this share,
    keep only those
    :return: tuple (kept_heads: list[str], head_to_terms: dict[str, list[str]])
    """
	# gather tokens (>=3 letters) and keep nouns
	counter = Counter()
	for m in cluster_meanings:
		for w in _tokenize_meaning_words(m):
			counter[w] += 1

	candidates = []
	for t, _freq in counter.most_common():
		if wn.synsets(t, pos=wn.NOUN):
			candidates.append(t)

	# term -> canonical head
	term_to_head, head_votes = {}, Counter()
	for t in candidates:
		h = _map_to_canonical_head(t)
		if h:
			term_to_head[t] = h
			head_votes[h] += counter[t]

	if not head_votes:
		return [], {}  # signal empty, caller must fallback to textual summary

	total = sum(head_votes.values())
	head_scores = {h: head_votes[h] / total for h in head_votes}

	# sort heads by score
	ordered = sorted(head_scores.items(), key=lambda x: -x[1])

	# dominance logic:

	if len(ordered) >= 2 and (
			ordered[0][1] + ordered[1][1]) >= dominance_pair_cover:
		# 1) if top-2 cover a large majority, show only them
		kept = [ordered[0][0], ordered[1][0]]
	else:
		# 2) otherwise, keep all with share >= min_head_share, capped by top_k
		kept = [h for h, sc in ordered if sc >= min_head_share][:top_k_heads]

	# ensure at most top_k_heads
	kept = kept[:top_k_heads] if top_k_heads else kept

	# head -> terms that voted for it (for transparency)
	head_to_terms = defaultdict(list)
	for t, h in term_to_head.items():
		if h in kept:
			head_to_terms[h].append(t)

	return kept, dict(head_to_terms)


def _hypernym_chain_hits(syn, target_syn, max_hops=5):
	"""
    check whether a synset reaches a target synset via hypernyms within k hops
    :param syn: starting wordnet synset
    :param target_syn: target wordnet synset to reach via hypernyms
    :param max_hops: maximum hypernym steps to search
    :return: bool indicating reachability within the hop limit
    """
	frontier, seen = [syn], set([syn])
	for _ in range(max_hops):
		nxt = []
		for s in frontier:
			if s == target_syn:
				return True
			for h in s.hypernyms():
				if h not in seen:
					seen.add(h);
					nxt.append(h)
		if not nxt: break
		frontier = nxt
	return False


def _map_to_canonical_head(term: str):
	"""
    map a token to one of the canonical heads using aliases and wordnet.
    :param term: candidate token (lower/upper case accepted)
    :return: canonical head string if mapped, otherwise lemma of first synset
    """
	# 1) aliases (exact/contains)
	lt = term.lower()
	if lt in _ALIAS_TO_HEAD:
		return _ALIAS_TO_HEAD[lt]
	for alias, head in _ALIAS_TO_HEAD.items():
		if alias in lt or head in lt:
			return head
	# 2) wordnet: does any noun sense climb to a canonical head?
	syns = wn.synsets(term, pos=wn.NOUN)
	for syn in syns[:3]:  # first few senses
		for head, head_syn in _CANON_HEADS.items():
			if _hypernym_chain_hits(syn, head_syn):
				return head

	# return the lemma of the first synset if present
	return syns[0].name().split(".")[0]


def _tokenize_meaning_words(meaning: str):
	"""
    split a meaning string into lowercase words (≥3 letters), minus stop-words.
    :param meaning: raw english meaning text
    :return: list[str] of cleaned tokens
    """
	words = re.findall(r'\b[a-zA-Z]{3,}\b', meaning.lower())
	return [w for w in words if w not in _WORD_STOP]


def _top_meaningful_terms(cluster_meanings, top_n=10, min_count=1):
	"""
    extract terms across meanings, sorted by frequency then alphabetic
    :param cluster_meanings: list[str] meanings to analyze
    :param top_n: maximum number of terms to return (None for all)
    :param min_count: minimum frequency for a term to be included
    :return: list[str] of meaningful terms
    """
	counter = Counter()
	for m in cluster_meanings:
		counter.update(_tokenize_meaning_words(m))
	# keep all terms meeting min_count, sorted by (freq desc, then alpha)
	terms = sorted(
		[t for t, c in counter.items() if c >= min_count],
		key=lambda t: (-counter[t], t)
	)
	return terms[:top_n]


def _pick_wordnet_pos(word: str):
	"""
    guess a useful wordnet part-of-speech for a word.
    prefers noun, then verb, adjective, adverb.
    :param word: token to test in wordnet
    :return: wn.NOUN/wn.VERB/wn.ADJ/wn.ADV or None if nothing found
    """
	# most of our themes are 'nouns' so that is the order
	order = [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]
	for pos in order:
		if wn.synsets(word, pos=pos):
			return pos
	return None


def _hypernyms_for_term(term: str, max_synsets=1, max_hypernyms=5):
	"""
    collect up to max_hypernyms hypernym lemma names for a term
    strategy: pick up to max_synsets synsets for the guessed pos and take
    their direct hypernyms, fallback to a mid-path slice when none exist
    :param term: token to look up in wordnet
    :param max_synsets: number of synsets to consider
    :param max_hypernyms: maximum hypernym lemma names to return
    :return: list[str] hypernym lemma names (may be empty)
    """
	pos = _pick_wordnet_pos(term)
	if not pos:
		return []
	syns = wn.synsets(term, pos=pos)[:max_synsets]
	hypers = []
	for s in syns:
		for h in s.hypernyms():
			hypers.extend([l.replace('_', ' ') for l in h.lemma_names()])

	path = syns[0].hypernym_paths()
	path = path[0][min(max_hypernyms, len(path[0]) // 2):]

	return hypers[:max_hypernyms] if hypers else path


def _theme_string_from_terms(terms, max_terms_to_show=None):
	"""
    join theme terms for display and optionally cap how many to show
    :param terms: list[str] of terms
    :param max_terms_to_show: optional cap on the number of terms
    :return: comma-separated theme string (may be empty)
    """
	if not terms:
		return ""
	if max_terms_to_show is not None:
		terms = terms[:max_terms_to_show]
	return ", ".join(terms)


def _hdbscan_cluster_embeddings(embeddings, metric='cosine',
                                min_samples=3, assign_noise_threshold=0.6):
	"""
    cluster sentence embeddings with hdbscan and optionally reassign noise
    normalization is applied for cosine metric, noise points may be attached to
    the nearest cluster when similarity passes the threshold
    :param embeddings: np.ndarray of shape (n_samples, dim)
    :param metric: similarity metric to guide normalization ('cosine' or other)
    :param min_samples: hdbscan min_samples parameter
    :param assign_noise_threshold: similarity threshold to attach noise to a
    cluster, set to None to keep all noise as -1
    :return: tuple (labels: np.ndarray, cluster_info: dict)
    """
	if len(embeddings) == 0:
		return np.array([]), {}

	# normalize embeddings for cosine metric
	if metric == 'cosine':
		emb = normalize(embeddings, norm='l2')
	else:
		emb = embeddings

	db = HDBSCAN(min_cluster_size=4, min_samples=min_samples,
	             metric='euclidean', prediction_data=True)

	labels = db.fit_predict(emb)  # -1 means noise

	# build cluster centroids for non-noise clusters
	clusters = defaultdict(list)
	for i, lab in enumerate(labels):
		if lab == -1:
			continue
		clusters[lab].append(emb[i])

	cluster_centers = {}
	for lab, vecs in clusters.items():
		cluster_centers[lab] = np.mean(np.vstack(vecs), axis=0)

	# Optionally assign noise points to nearest cluster
	# only if similarity >= threshold
	if assign_noise_threshold is not None and len(cluster_centers) > 0:
		for i, lab in enumerate(labels):
			if lab != -1:
				continue  # already assigned
			# compute similarity to each cluster centroid
			sims = {c_lab: cosine_similarity(emb[i].reshape(1, -1),
			                                 center.reshape(1, -1))[0, 0]
			        for c_lab, center in cluster_centers.items()}
			best_lab, best_sim = max(sims.items(), key=lambda x: x[1])
			if best_sim >= assign_noise_threshold:
				labels[i] = best_lab

	cluster_info = {
		'n_clusters'     : len(cluster_centers),
		'labels'         : labels,
		'cluster_centers': cluster_centers,
		'noise_count'    : int(np.sum(labels == -1))
	}
	return labels, cluster_info


def _analyze_cluster_radicals(cluster_characters, char_to_radicals):
	"""
    summarize radical distribution within a cluster of characters
    :param cluster_characters: list[str] characters in the semantic cluster
    :param char_to_radicals: dict mapping character -> set/list of radicals
    :return: dict with 'top_radicals', 'radical_diversity',
    'dominant_radical_percentage', 'total_unique_radicals'
    """
	if not cluster_characters or not char_to_radicals:
		return {'top_radicals': [], 'radical_diversity': 0}

	radical_counts = Counter()
	chars_with_radicals = 0

	for char in cluster_characters:
		if char in char_to_radicals:
			radicals = char_to_radicals[char]
			radical_counts.update(radicals)
			chars_with_radicals += 1

	if not radical_counts:
		return {'top_radicals': [], 'radical_diversity': 0}

	top_radicals = radical_counts.most_common(5)
	radical_diversity = len(radical_counts) / max(chars_with_radicals, 1)

	# calculate dominance of top radical
	if top_radicals:
		dominant_radical_percentage = \
			(top_radicals[0][1] / chars_with_radicals) * 100
	else:
		dominant_radical_percentage = 0

	return {
		'top_radicals'               : top_radicals,
		'radical_diversity'          : radical_diversity,
		'dominant_radical_percentage': dominant_radical_percentage,
		'total_unique_radicals'      : len(radical_counts)
	}


def _calculate_coherence_score(similarity_matrix):
	"""
    compute a simple coherence score from pairwise similarities
    higher mean and lower variance → higher score, bounded below by 0
    :param similarity_matrix: square np.ndarray of cosine similarities
    :return: float coherence value in [0, 1]
    """
	# remove diagonal
	mask = np.eye(len(similarity_matrix), dtype=bool)
	similarities = similarity_matrix[~mask]

	# higher coherence = higher mean similarity+lower variance
	mean_sim = np.mean(similarities)
	std_sim = np.std(similarities)

	# normalize to 0-1 range
	coherence = mean_sim * (1 - std_sim)
	return max(0, coherence)


def _find_optimal_clusters_kmeans(embeddings, max_clusters=15,
                                  min_clusters=2):
	"""
    pick a k for k-means using silhouette score on embeddings
    :param embeddings: np.ndarray of shape (n_samples, dim)
    :param max_clusters: maximum k to consider
    :param min_clusters: minimum k to consider
    :return: int k (or tuple (k, fitted_kmeans) in this implementation)
    """
	if len(embeddings) < min_clusters:
		return 1

	max_clusters = min(max_clusters, len(embeddings) - 1)
	if max_clusters < min_clusters:
		return 1

	best_score = -1
	best_k = min_clusters
	best_kmeans = None

	for k in range(min_clusters, max_clusters + 1):
		try:
			kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
			labels = kmeans.fit_predict(embeddings)
			score = silhouette_score(embeddings, labels)

			if score > best_score:
				best_score = score
				best_k = k
				best_kmeans = kmeans
		except:
			# skip if clustering fails for this k
			continue

	return best_k, best_kmeans


def _generate_cluster_summary(cluster_meanings):
	"""
    build a readable theme string from meaningful terms across meanings
    if no such terms exist, fall back to the most common raw meaning
    :param cluster_meanings: list[str] meanings in the cluster
    :return: string theme label
    """
	terms = _top_meaningful_terms(cluster_meanings, top_n=20, min_count=1)
	theme_str = _theme_string_from_terms(terms) if terms else \
		(Counter(cluster_meanings).most_common(1)[0][0].split(',')[0].strip()
		 if cluster_meanings else "")
	return theme_str


def _extract_themes_with_cluster_details(meanings, characters,
                                         embeddings, kmeans,
                                         optimal_clusters, max_themes,
                                         char_to_radicals):
	"""
    derive readable themes and rich per-cluster details (incl. radicals)
    works for both hdbscan pseudo kmeans and real kmeans objects
    :param meanings: list[str] meanings (aligned to embeddings)
    :param characters: optional list[str] corresponding characters (or None)
    :param embeddings: np.ndarray sentence embeddings
    :param kmeans: fitted object exposing labels_ (and optionally centers)
    :param optimal_clusters: number of clusters implied by labels
    :param max_themes: cap on how many cluster themes to return (ordered)
    :param char_to_radicals: optional dict character -> radicals set/list
    :return: tuple (themes: list[str], cluster_details: list[dict])
    """
	if len(meanings) < 3:
		# trivial case
		simple = meanings[:max_themes]
		# build a minimal 'details' with hypernyms anyway
		term_hyper = {t: _hypernyms_for_term(t) for t in simple}
		return simple, [{
			'cluster_id'      : 0,
			'theme'           : _theme_string_from_terms(simple),
			'theme_terms'     : simple,
			'term_hypernyms'  : term_hyper,
			'size'            : len(simple),
			'meanings'        : meanings,
			'characters'      : characters or [],
			'radical_analysis': _analyze_cluster_radicals(characters or [],
			                                              char_to_radicals)
		}]

	themes = []
	cluster_details = []
	labels = kmeans.labels_

	for i in range(optimal_clusters):
		cluster_indices = np.where(labels == i)[0]
		if len(cluster_indices) == 0:
			continue

		cluster_meanings = [meanings[int(idx)] for idx in cluster_indices]
		cluster_chars = [characters[int(idx)] for idx in cluster_indices] \
			if characters else []

		# radicals
		radical_analysis = _analyze_cluster_radicals(cluster_chars,
		                                             char_to_radicals)

		heads, head_to_terms = _semantic_heads_from_meanings(
			cluster_meanings,
			top_k_heads=2,
			min_head_share=0.12,  # changed to diff values for best results
			dominance_pair_cover=0.70,
			# if top-2 ≥ 70% of mapped terms, keep only those
		)

		if heads:
			theme_str = ", ".join(heads)
			# pretty hypernyms for canonical heads (controlled)
			term_hypernyms = \
				{h:
					 _CANON_HEAD_DISPLAY_HYPERNYMS.get(h,
					                                   _hypernyms_for_term(h))
				 for h in heads}
		else:
			# robust fallback if mapping yielded nothing:
			# 1) short textual summary from old function
			theme_str = _generate_cluster_summary(cluster_meanings)
			# 2) also provide top frequent nouns
			fallback_terms = [w for w, _ in Counter(
				sum([_tokenize_meaning_words(m) for m in cluster_meanings],
				    [])
			).most_common(10)]
			heads = fallback_terms
			term_hypernyms = {t: _hypernyms_for_term(t) for t in heads}

		cluster_detail = {
			'cluster_id'      : i,
			'theme'           : theme_str,
			'theme_terms'     : heads,
			'head_members'    : head_to_terms,
			'term_hypernyms'  : term_hypernyms,
			'size'            : len(cluster_indices),
			'meanings'        : cluster_meanings,
			'characters'      : cluster_chars,
			'radical_analysis': radical_analysis
		}

		cluster_details.append(cluster_detail)
		themes.append(
			{'theme': theme_str, 'cluster_size': len(cluster_indices)})

	# sort by cluster size
	themes.sort(key=lambda x: x['cluster_size'], reverse=True)
	cluster_details.sort(key=lambda x: x['size'], reverse=True)

	# return the readable themes (ordered) + the rich details (with full
	# lists and hypernyms)
	return [t['theme'] for t in themes[:max_themes]], cluster_details


class HanziSemanticAnalyzer:
	def __init__(self, model_name='all-MiniLM-L6-v2'):
		"""
        initialize a sentence transformer used to embed meaning strings
        :param model_name: huggingface model id to load for embeddings
        :return:
        """
		self.model = SentenceTransformer(model_name)

	def analyze_community_semantic_coherence(self, community_meanings):
		"""
        compute semantic coherence for meanings within one community
        returns average pairwise similarity and a simple coherence score.
        :param community_meanings: list[str] meanings for the community
        :return: dict with 'avg_similarity', 'coherence_score', and matrix
        """
		if len(community_meanings) < 2:
			return {
				'avg_similarity'   : 1.0,
				'coherence_score'  : 1.0,
				'semantic_clusters': 1,
				'dominant_themes'  : []
			}

		# generate embeddings for all meanings
		embeddings = self.model.encode(community_meanings)

		# calc pairwise similarities
		similarity_matrix = cosine_similarity(embeddings)

		# remove diagonal (self-similarity)
		mask = np.eye(len(similarity_matrix), dtype=bool)
		similarities = similarity_matrix[~mask]

		avg_similarity = np.mean(similarities)
		coherence_score = _calculate_coherence_score(similarity_matrix)

		return {
			'avg_similarity'   : avg_similarity,
			'coherence_score'  : coherence_score,
			'similarity_matrix': similarity_matrix
		}

	def analyze_community_semantic_coherence_with_radicals(self,
	                                                       community_meanings,
	                                                       community_characters=None,
	                                                       char_to_radicals=None):
		"""
        compute coherence as above + cluster themes and radical summaries
        hdbscan is used to form semantic clusters and noise may be reassigned
        :param community_meanings: list[str] meanings in the community
        :param community_characters: optional list[str] aligned characters
        :param char_to_radicals: optional dict character -> radicals set/list
        :return: dict with similarities, coherence, cluster count, themes,
        similarity_matrix, and rich 'semantic_cluster_details'
        """
		if len(community_meanings) < 2:
			return {
				'avg_similarity'          : 1.0,
				'coherence_score'         : 1.0,
				'semantic_clusters'       : 1,
				'dominant_themes'         : [],
				'semantic_cluster_details': []
			}

		# generate embeddings for all meanings
		embeddings = self.model.encode(community_meanings,
		                               convert_to_numpy=True)

		# choose DBSCAN or KMeans #todo FLAG FOR US TO CHANGE
		use_dbscan = True
		if use_dbscan:
			labels, cluster_info = _hdbscan_cluster_embeddings(
				embeddings, metric='cosine', min_samples=4
			)

			# remap DBSCAN labels to 0..k-1, ignore noise (-1)
			unique_labels = sorted([l for l in set(labels) if l != -1])
			label_map = {orig: i for i, orig in enumerate(unique_labels)}
			remapped_labels = np.array(
				[label_map[l] if l != -1 else -1 for l in labels]
			)

			centers_ordered = [
				cluster_info['cluster_centers'][lab] for lab in
				unique_labels
			]

			# dummy "kmeans-like" object to keep the rest of code more generic
			kmeans = type("K", (), {
				"labels_"         : remapped_labels,
				"cluster_centers_": np.vstack(
					centers_ordered) if centers_ordered else np.zeros(
					(0, embeddings.shape[1]))
			})()

			semantic_clusters = len(unique_labels)

		else:
			semantic_clusters, kmeans = _find_optimal_clusters_kmeans(
				embeddings,
				max_clusters=min(15, len(community_meanings) // 2)
			)

		# calc pairwise similarities
		similarity_matrix = cosine_similarity(embeddings)
		mask = np.eye(len(similarity_matrix), dtype=bool)
		similarities = similarity_matrix[~mask]
		avg_similarity = np.mean(similarities)
		coherence_score = _calculate_coherence_score(similarity_matrix)

		# extract dominant themes (works for both DBSCAN dummy and KMeans)
		# and cluster details
		dominant_themes, cluster_details = \
			_extract_themes_with_cluster_details(community_meanings,
			                                     community_characters,
			                                     embeddings, kmeans,
			                                     semantic_clusters, 15,
			                                     char_to_radicals)

		return {
			'avg_similarity'          : avg_similarity,
			'coherence_score'         : coherence_score,
			'semantic_clusters'       : semantic_clusters,
			'dominant_themes'         : dominant_themes,
			'similarity_matrix'       : similarity_matrix,
			'semantic_cluster_details': cluster_details
		}
