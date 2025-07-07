// TextFormatter.js - Handles text formatting, markdown parsing, and syntax highlighting
export class TextFormatter {
    constructor() {
        this.syntaxHighlightingEnabled = false;
        this.bionicMode = false;
        this.focusGradient = false;
    }

    parseMarkdown(text) {
        try {
            let html = text;
            
            // Escape HTML to prevent XSS
            html = html.replace(/&/g, '&amp;')
                      .replace(/</g, '&lt;')
                      .replace(/>/g, '&gt;');
            
            // Code blocks FIRST
            html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                const language = lang ? lang.toLowerCase() : '';
                return `<pre><code class="language-${language}">${code.trim()}</code></pre>`;
            });
            
            // Inline code
            html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
            
            // Headers
            html = html.replace(/^#{6}\s+(.+)$/gm, '<h6>$1</h6>');
            html = html.replace(/^#{5}\s+(.+)$/gm, '<h5>$1</h5>');
            html = html.replace(/^#{4}\s+(.+)$/gm, '<h4>$1</h4>');
            html = html.replace(/^#{3}\s+(.+)$/gm, '<h3>$1</h3>');
            html = html.replace(/^#{2}\s+(.+)$/gm, '<h2>$1</h2>');
            html = html.replace(/^#{1}\s+(.+)$/gm, '<h1>$1</h1>');
            
            // Bold and italic
            html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
            html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
            html = html.replace(/\*([^*\n]+?)\*/g, '<em>$1</em>');
            html = html.replace(/_([^_\n]+?)_/g, '<em>$1</em>');
            
            // Strikethrough
            html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');
            
            // Links
            html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
            
            // Blockquotes
            html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
            
            // Horizontal rules
            html = html.replace(/^(---|\*\*\*|___)$/gm, '<hr>');
            
            // Lists
            html = html.replace(/^[\*\-]\s+(.+)$/gm, '<li>$1</li>');
            html = html.replace(/(<li>.*<\/li>\s*)+/g, (match) => {
                return '<ul>' + match + '</ul>';
            });
            
            // Ordered lists
            html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
            
            // Line breaks
            html = html.replace(/\n\n/g, '</p><p>');
            html = html.replace(/\n/g, '<br>');
            
            // Wrap in paragraphs
            if (!html.startsWith('<')) {
                html = '<p>' + html + '</p>';
            }
            
            // Clean up empty paragraphs
            html = html.replace(/<p><\/p>/g, '');
            html = html.replace(/<p>(<[^>]+>)/g, '$1');
            html = html.replace(/(<\/[^>]+>)<\/p>/g, '$1');
            
            return html;
        } catch (e) {
            console.error('Error parsing markdown:', e);
            return text;
        }
    }

    applyBionicReading(text) {
        try {
            return text.split(/\s+/).map(word => {
                if (!word) return '';
                if (word.length <= 3) {
                    return `<span class="word"><span class="bold">${word}</span></span>`;
                }
                const boldLength = Math.ceil(word.length * 0.4);
                const boldPart = word.slice(0, boldLength);
                const normalPart = word.slice(boldLength);
                return `<span class="word"><span class="bold">${boldPart}</span>${normalPart}</span>`;
            }).join(' ');
        } catch (e) {
            console.error('Error applying bionic reading:', e);
            return text;
        }
    }

    highlightSyntax(text, language = '') {
        if (!this.syntaxHighlightingEnabled) return text;
        
        if (language) {
            return this.highlightCode(text, language);
        } else {
            return this.highlightNaturalLanguage(text);
        }
    }

    highlightCode(code, language) {
        const languages = {
            javascript: {
                keywords: /\b(function|var|let|const|if|else|for|while|return|class|extends|import|export|async|await|try|catch|throw|new|this|super)\b/g,
                strings: /(["'`])(?:(?=(\\?))\2[\s\S])*?\1/g,
                comments: /(\/\/.*$|\/\*[\s\S]*?\*\/)/gm,
                functions: /\b(\w+)(?=\()/g,
                numbers: /\b(\d+\.?\d*)\b/g,
                operators: /([+\-*/%=<>!&|?:]+)/g
            },
            python: {
                keywords: /\b(def|class|if|elif|else|for|while|return|import|from|as|try|except|finally|with|lambda|yield|break|continue|pass|raise|assert|del|global|nonlocal|and|or|not|in|is)\b/g,
                strings: /(["'])((?:\\\1|(?:(?!\1).))*)(\1)/g,
                comments: /(#.*$)/gm,
                functions: /\b(\w+)(?=\()/g,
                numbers: /\b(\d+\.?\d*)\b/g,
                decorators: /(@\w+)/g,
                self: /\b(self|cls)\b/g
            },
            sql: {
                keywords: /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP BY|ORDER BY|HAVING|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TABLE|DATABASE|INDEX|VIEW|AS|AND|OR|NOT|IN|EXISTS|BETWEEN|LIKE|LIMIT|OFFSET)\b/gi,
                strings: /(["'])(?:(?=(\\?))\2[\s\S])*?\1/g,
                comments: /(--.*$|\/\*[\s\S]*?\*\/)/gm,
                functions: /\b(COUNT|SUM|AVG|MAX|MIN|UPPER|LOWER|SUBSTRING|CONCAT)\b/gi,
                numbers: /\b(\d+\.?\d*)\b/g
            }
        };

        let highlighted = this._escapeHtml(code);
        const lang = languages[language.toLowerCase()];
        
        if (lang) {
            // Apply syntax highlighting in specific order
            if (lang.strings) highlighted = highlighted.replace(lang.strings, '<span class="syntax-string">$&</span>');
            if (lang.comments) highlighted = highlighted.replace(lang.comments, '<span class="syntax-comment">$&</span>');
            if (lang.keywords) highlighted = highlighted.replace(lang.keywords, '<span class="syntax-keyword">$&</span>');
            if (lang.functions) highlighted = highlighted.replace(lang.functions, '<span class="syntax-function">$1</span>');
            if (lang.numbers) highlighted = highlighted.replace(lang.numbers, '<span class="syntax-number">$&</span>');
            if (lang.decorators) highlighted = highlighted.replace(lang.decorators, '<span class="syntax-decorator">$&</span>');
            if (lang.self) highlighted = highlighted.replace(lang.self, '<span class="syntax-self">$&</span>');
            if (lang.operators) highlighted = highlighted.replace(lang.operators, '<span class="syntax-operator">$&</span>');
        }
        
        return highlighted;
    }

    highlightNaturalLanguage(text) {
        const patterns = {
            adjectives: /\b(quick|brown|lazy|big|small|beautiful|ugly|fast|slow|happy|sad|good|bad|new|old|young|first|last|long|short|high|low|right|wrong|large|tiny|massive|huge|red|blue|green|yellow|black|white|bright|dark|hot|cold|warm|cool)\b/gi,
            adverbs: /\b(quickly|slowly|carefully|happily|sadly|really|very|extremely|quite|rather|too|almost|nearly|just|already|still|always|never|often|rarely|sometimes|usually|suddenly|immediately|eventually|finally|recently|soon|today|tomorrow|yesterday|now|then|here|there)\b/gi,
            verbs: /\b(is|are|was|were|been|be|have|has|had|do|does|did|will|would|shall|should|may|might|can|could|must|go|goes|went|come|came|see|saw|seen|make|made|take|took|taken|give|gave|given|find|found|think|thought|tell|told|become|became|leave|left|feel|felt|bring|brought|begin|began|keep|kept|hold|held|write|wrote|written|stand|stood|hear|heard|let|run|ran|mean|meant|set|meet|met|pay|paid|sit|sat|speak|spoke|spoken|lie|lay|lead|led|read|grow|grew|grown|lose|lost|send|sent|build|built|understand|understood|draw|drew|drawn|break|broke|broken|spend|spent|cut|drive|drove|driven|eat|ate|eaten|fall|fell|fallen|choose|chose|chosen|sleep|slept|win|won|wear|wore|worn|sell|sold|throw|threw|thrown|catch|caught|buy|bought|fight|fought|teach|taught|reach|reached|jump|jumps|jumped|play|plays|played|work|works|worked|live|lives|lived|move|moves|moved|stop|stops|stopped|carry|carries|carried|open|opens|opened|watch|watches|watched|follow|follows|followed|create|creates|created|speak|speaks|add|adds|added|grow|grows|offer|offers|offered|remember|remembers|remembered|love|loves|loved|consider|considers|considered|appear|appears|appeared|buy|buys|serve|serves|served|die|dies|died|send|sends)\b/gi,
            nouns: /\b(time|person|people|year|years|way|day|days|man|men|woman|women|thing|things|child|children|world|life|hand|hands|part|parts|place|places|week|weeks|case|cases|point|points|government|company|companies|number|numbers|group|groups|problem|problems|fact|facts|money|question|questions|work|night|nights|home|water|room|rooms|mother|father|area|areas|story|stories|month|months|right|rights|study|studies|book|books|eye|eyes|job|jobs|word|words|business|issue|issues|side|sides|kind|kinds|head|heads|house|houses|service|services|friend|friends|hour|hours|game|games|line|lines|moment|moments|member|members|law|laws|car|cars|city|cities|community|communities|name|names|president|team|teams|minute|minutes|idea|ideas|body|bodies|kid|kids|table|tables|information|back|parent|parents|face|faces|level|levels|office|door|doors|health|person|art|war|history|party|parties|result|results|change|changes|morning|mornings|reason|reasons|research|girl|girls|guy|guys|food|moment|air|teacher|teachers|force|forces|foot|feet|boy|boys|age|ages|policy|policies|process|processes|music|market|markets|sense|senses|nation|plan|plans|college|interest|interests|death|deaths|experience|experiences|effect|effects|use|uses|class|classes|control|field|fields|development|developments|ability|abilities|staff|step|steps|dog|dogs|fund|funds|town|towns|road|roads|project|projects|test|tests|decision|decisions|method|methods|record|records|thought|thoughts|department|departments|management|church|churches|view|views|relationship|relationships|dog|dogs|cat|cats|tree|trees|river|rivers|mountain|mountains|rain|snow|sun|moon|star|stars|cloud|clouds|flower|flowers|bird|birds|fish|ocean|sea|lake|forest|garden|park|street|building|buildings|store|stores|restaurant|restaurants|hotel|hotels|airport|airports|station|stations|hospital|hospitals|bank|banks|machine|machines|device|devices|computer|computers|phone|phones|internet|website|software|app|apps|tool|tools|table|chair|chairs|desk|desks|bed|beds|window|windows|kitchen|bedroom|bedrooms|bathroom|bathrooms|living|dining|garage|yard|fence|neighbor|neighbors|village|villages|county|counties|state|states|country|countries|continent|continents|planet|animal|animals|plant|plants|food|drink|drinks|meal|meals|breakfast|lunch|dinner|coffee|tea|juice|bread|meat|vegetable|vegetables|fruit|fruits|apple|apples|orange|oranges|chicken|beef|pork|salad|soup|sandwich|sandwiches|pizza|pasta|rice|egg|eggs|cheese|milk|butter|sugar|salt|pepper|oil|sauce|dessert|desserts|cake|cakes|cookie|cookies|ice|cream|chocolate|candy|candies|doctor|doctors|nurse|nurses|patient|patients|medicine|medicines|pill|pills|surgery|surgeries|disease|diseases|pain|cure|cures|treatment|treatments|exercise|exercises|sport|sports|football|basketball|baseball|soccer|tennis|golf|running|swimming|cycling|hiking|camping|fishing|hunting|dancing|singing|painting|drawing|reading|writing|movie|movies|show|shows|song|songs|band|bands|artist|artists|actor|actors|actress|actresses|director|directors|producer|producers|camera|cameras|picture|pictures|photo|photos|video|videos|color|colors|shape|shapes|size|sizes|weight|height|length|width|depth|speed|temperature|pressure|energy|power|light|sound|heat|cold|wind|storm|storms|earthquake|earthquakes|volcano|volcanoes|disaster|disasters|accident|accidents|danger|dangers|safety|risk|risks|chance|chances|opportunity|opportunities|choice|choices|option|options|solution|solutions|answer|answers|problem|mistake|mistakes|error|errors|success|successes|failure|failures|win|loss|losses|victory|victories|defeat|defeats|prize|prizes|award|awards|gift|gifts|present|presents|surprise|surprises|secret|secrets|truth|lie|lies|joke|jokes|humor|fun|entertainment|hobby|hobbies|interest|toy|toys|doll|dolls|ball|balls|puzzle|puzzles|card|cards|board|boards|dice|coin|coins|ticket|tickets|price|prices|cost|costs|value|values|worth|sale|sales|discount|discounts|deal|deals|offer|shop|shopping|customer|customers|client|clients|guest|guests|visitor|visitors|tourist|tourists|trip|trips|journey|journeys|travel|destination|destinations|map|maps|direction|directions|distance|distances|mile|miles|kilometer|kilometers|inch|inches|foot|centimeter|centimeters|meter|meters|second|seconds|minute|hour|day|week|month|year|decade|decades|century|centuries|morning|afternoon|evening|night|dawn|sunrise|sunset|midnight|season|seasons|spring|summer|autumn|fall|winter|weather|climate|nature|environment|planet|universe|space|gravity|matter|atom|atoms|element|elements|chemical|chemicals|reaction|reactions|experiment|experiments|science|scientist|scientists|research|discovery|discoveries|invention|inventions|technology|technologies|progress|future|past|present|memory|memories|mind|minds|brain|brains|thought|dream|dreams|imagination|imaginations|creativity|inspiration|motivation|emotion|emotions|feeling|feelings|mood|moods|happiness|sadness|anger|fear|fears|worry|worries|stress|stresses|relief|comfort|joy|joys|pleasure|pleasures|satisfaction|disappointment|disappointments|surprise|excitement|excitements|interest|curiosity|curiosities|wonder|wonders|beauty|beauties|art|culture|cultures|tradition|traditions|custom|customs|habit|habits|behavior|behaviors|attitude|attitudes|personality|personalities|character|characters|quality|qualities|skill|skills|talent|talents|strength|strengths|weakness|weaknesses|advantage|advantages|disadvantage|disadvantages|benefit|benefits|harm|harms|damage|damages|injury|injuries|wound|wounds|scar|scars|mark|marks|spot|spots|dot|dots|line|circle|circles|square|squares|triangle|triangles|rectangle|rectangles|pattern|patterns|design|designs|style|styles|fashion|fashions|trend|trends|brand|brands|model|models|version|versions|type|types|category|categories|class|grade|grades|rank|ranks|position|positions|status|statuses|role|roles|duty|duties|task|tasks|mission|missions|goal|goals|target|targets|objective|objectives|purpose|purposes|reason|cause|causes|source|sources|origin|origins|root|roots|base|bases|foundation|foundations|ground|grounds|floor|floors|ceiling|ceilings|wall|walls|corner|corners|edge|edges|surface|surfaces|material|materials|substance|substances|liquid|liquids|solid|solids|gas|gases|air|fire|flame|flames|smoke|dust|dirt|mud|stone|stones|rock|rocks|sand|glass|metal|metals|gold|silver|iron|steel|copper|wood|paper|plastic|rubber|leather|cloth|fabric|fabrics|thread|threads|rope|ropes|chain|chains|wire|wires|nail|nails|screw|screws|bolt|bolts|lock|locks|key|keys|handle|handles|button|buttons|switch|switches|plug|plugs|battery|batteries|engine|engines|motor|motors|wheel|wheels|tire|tires|brake|brakes|gear|gears|belt|belts|pipe|pipes|tube|tubes|hole|holes|gap|gaps|crack|cracks|break|cut|tear|tears|fold|folds|bend|bends|twist|twists|turn|turns|spin|spins|roll|rolls|slide|slides|slip|slips|fall|drop|drops|rise|rises|lift|lifts|push|pushes|pull|pulls|press|presses|squeeze|squeezes|stretch|stretches|compress|compresses|expand|expands|grow|shrink|shrinks|increase|increases|decrease|decreases|change|move|movement|movements|action|actions|activity|activities|event|events|incident|incidents|situation|situations|condition|conditions|state|stage|stages|phase|phases|period|periods|era|eras|age|time|moment|instant|instants|occasion|occasions|instance|instances|example|examples|sample|samples|piece|pieces|bit|bits|particle|particles|drop|grain|grains|pile|piles|heap|heaps|stack|stacks|bunch|bunches|bundle|bundles|package|packages|box|boxes|bag|bags|bottle|bottles|jar|jars|can|cans|cup|cups|glass|glasses|plate|plates|bowl|bowls|dish|dishes|pot|pots|pan|pans|spoon|spoons|fork|forks|knife|knives|tool|instrument|instruments|equipment|device|gadget|gadgets|appliance|appliances|furniture|fixture|fixtures|decoration|decorations|ornament|ornaments|jewelry|jewel|jewels|treasure|treasures|collection|collections|set|sets|kit|kits|pack|packs|group|team|crew|crews|gang|gangs|band|crowd|crowds|audience|audiences|population|populations|society|societies|civilization|civilizations|culture|community|family|families|parent|child|baby|babies|infant|infants|adult|adults|teenager|teenagers|youth|youths|elder|elders|ancestor|ancestors|descendant|descendants|relative|relatives|relation|relations|connection|connections|link|links|bond|bonds|tie|ties|relationship|friendship|friendships|love|romance|romances|marriage|marriages|wedding|weddings|divorce|divorces|birth|births|death|life|existence|existences|being|beings|creature|creatures|human|humans|person|individual|individuals|self|selves|identity|identities|name|title|titles|label|labels|tag|tags|sign|signs|signal|signals|symbol|symbols|mark|flag|flags|banner|banners|poster|posters|billboard|billboards|advertisement|advertisements|announcement|announcements|message|messages|letter|letters|note|notes|memo|memos|report|reports|document|documents|file|files|folder|folders|book|magazine|magazines|newspaper|newspapers|article|articles|story|chapter|chapters|page|pages|paragraph|paragraphs|sentence|sentences|phrase|phrases|word|term|terms|expression|expressions|saying|sayings|quote|quotes|statement|statements|declaration|declarations|announcement|comment|comments|remark|remarks|observation|observations|opinion|opinions|belief|beliefs|idea|concept|concepts|theory|theories|principle|principles|rule|rules|law|regulation|regulations|policy|guideline|guidelines|standard|standards|criterion|criteria|requirement|requirements|specification|specifications|instruction|instructions|direction|procedure|procedures|process|method|technique|techniques|strategy|strategies|plan|approach|approaches|way|manner|manners|style|mode|modes|form|forms|format|formats|structure|structures|system|systems|organization|organizations|arrangement|arrangements|order|orders|sequence|sequences|series|pattern|cycle|cycles|loop|loops|circuit|circuits|network|networks|web|webs|grid|grids|matrix|matrices|framework|frameworks|scheme|schemes|program|programs|project|schedule|schedules|agenda|agendas|calendar|calendars|date|dates|appointment|appointments|meeting|meetings|conference|conferences|session|sessions|class|course|courses|lesson|lessons|lecture|lectures|presentation|presentations|speech|speeches|talk|talks|conversation|conversations|discussion|discussions|debate|debates|argument|arguments|dispute|disputes|conflict|conflicts|fight|battle|battles|war|competition|competitions|contest|contests|race|races|match|matches|game|tournament|tournaments|championship|championships|league|leagues|season|round|rounds|level|stage|phase|period|quarter|quarters|half|halves|final|finals|result|score|scores|point|goal|target|victory|defeat|draw|draws|tie|win|loss|gain|gains|profit|profits|benefit|advantage|success|achievement|achievements|accomplishment|accomplishments|progress|improvement|improvements|development|growth|increase|decrease|decline|declines|reduction|reductions|rise|fall|change|difference|differences|variation|variations|variety|varieties|diversity|diversities|range|ranges|scope|scopes|extent|extents|degree|degrees|amount|amounts|quantity|quantities|volume|volumes|mass|masses|weight|measure|measures|unit|units|scale|scales|proportion|proportions|ratio|ratios|percentage|percentages|fraction|fractions|part|portion|portions|section|sections|segment|segments|division|divisions|category|class|type|kind|sort|sorts|form|version|model|brand|style|design|pattern|shape|size|color|texture|textures|quality|property|properties|characteristic|characteristics|feature|features|attribute|attributes|trait|traits|aspect|aspects|element|component|components|ingredient|ingredients|factor|factors|variable|variables|parameter|parameters|condition|criterion|standard|principle|rule|law|theory|concept|idea|belief|opinion|view|perspective|perspectives|angle|angles|approach|method|way|technique|strategy|plan|system|process|procedure|practice|practices|habit|custom|tradition|culture|behavior|action|activity|movement|motion|motions|gesture|gestures|expression|reaction|reactions|response|responses|reply|replies|answer|solution|result|outcome|outcomes|consequence|consequences|effect|impact|impacts|influence|influences|change|transformation|transformations|conversion|conversions|transition|transitions|shift|shifts|switch|switches|exchange|exchanges|trade|trades|deal|transaction|transactions|transfer|transfers|delivery|deliveries|shipment|shipments|transport|transports|travel|journey|trip|voyage|voyages|expedition|expeditions|adventure|adventures|exploration|explorations|discovery|finding|findings|observation|examination|examinations|inspection|inspections|investigation|investigations|inquiry|inquiries|search|searches|hunt|hunts|quest|quests|pursuit|pursuits|chase|chases|race|escape|escapes|flight|flights|run|walk|walks|step|pace|paces|speed|velocity|velocities|acceleration|accelerations|momentum|force|power|energy|strength|pressure|tension|tensions|stress|strain|strains|load|loads|burden|burdens|weight|mass|density|densities|concentration|concentrations|intensity|intensities|level|degree|extent|amount|quantity|volume|capacity|capacities|ability|capability|capabilities|potential|potentials|possibility|possibilities|opportunity|chance|risk|danger|threat|threats|hazard|hazards|obstacle|obstacles|barrier|barriers|challenge|challenges|difficulty|difficulties|problem|issue|matter|matters|concern|concerns|worry|fear|anxiety|anxieties|stress|pressure|tension|conflict|crisis|crises|emergency|emergencies|disaster|accident|incident|event|situation|circumstance|circumstances|condition|state|status|position|location|locations|place|spot|point|area|region|regions|zone|zones|district|districts|section|sector|sectors|territory|territories|domain|domains|field|realm|realms|sphere|spheres|scope|range|extent|limit|limits|boundary|boundaries|border|borders|edge|margin|margins|frame|frames|outline|outlines|shape|form|figure|figures|image|images|picture|photo|photograph|photographs|portrait|portraits|landscape|landscapes|scene|scenes|view|sight|sights|vision|visions|appearance|appearances|look|looks|style|fashion|trend|design|pattern|decoration|ornament|detail|details|feature|aspect|quality|texture|color|shade|shades|tone|tones|hue|hues|brightness|darkness|light|shadow|shadows|reflection|reflections|shine|shines|glow|glows|sparkle|sparkles|flash|flashes|beam|beams|ray|rays|wave|waves|particle|quantum|quanta|atom|molecule|molecules|cell|cells|tissue|tissues|organ|organs|system|body|skeleton|skeletons|bone|bones|muscle|muscles|nerve|nerves|blood|heart|hearts|lung|lungs|liver|livers|kidney|kidneys|stomach|stomachs|intestine|intestines|skin|hair|hairs|eye|ear|ears|nose|noses|mouth|mouths|tooth|teeth|tongue|tongues|lip|lips|face|head|neck|necks|shoulder|shoulders|arm|arms|elbow|elbows|wrist|wrists|hand|finger|fingers|thumb|thumbs|chest|chests|breast|breasts|back|waist|waists|hip|hips|leg|legs|knee|knees|ankle|ankles|foot|toe|toes|nail|skin|flesh|blood|sweat|tear|saliva|mucus|pus|urine|feces|vomit|bile|hormone|hormones|enzyme|enzymes|protein|proteins|vitamin|vitamins|mineral|minerals|nutrient|nutrients|calorie|calories|fat|fats|carbohydrate|carbohydrates|sugar|starch|starches|fiber|fibers|cholesterol|acid|acids|base|bases|salt|compound|compounds|mixture|mixtures|solution|solutions|suspension|suspensions|emulsion|emulsions|colloid|colloids|crystal|crystals|powder|powders|dust|smoke|vapor|vapors|steam|mist|mists|fog|fogs|cloud|rain|snow|ice|frost|dew|hail|sleet|storm|thunder|lightning|wind|breeze|breezes|gale|gales|hurricane|hurricanes|tornado|tornadoes|cyclone|cyclones|typhoon|typhoons|flood|floods|drought|droughts|earthquake|volcano|eruption|eruptions|lava|magma|ash|ashes|rock|stone|boulder|boulders|pebble|pebbles|sand|gravel|soil|dirt|mud|clay|dust|mineral|ore|ores|gem|gems|diamond|diamonds|ruby|rubies|emerald|emeralds|sapphire|sapphires|pearl|pearls|gold|silver|platinum|copper|iron|steel|aluminum|zinc|lead|tin|nickel|chrome|titanium|uranium|coal|oil|gas|petroleum|fuel|fuels|gasoline|diesel|kerosene|propane|methane|ethane|butane|hydrogen|oxygen|nitrogen|carbon|sulfur|phosphorus|chlorine|fluorine|helium|neon|argon|sodium|potassium|calcium|magnesium|silicon|glass|ceramic|ceramics|plastic|rubber|polymer|polymers|resin|resins|adhesive|adhesives|glue|paste|pastes|cement|concrete|mortar|plaster|paint|paints|varnish|varnishes|lacquer|lacquers|enamel|enamels|coating|coatings|finish|finishes|polish|polishes|wax|waxes|oil|grease|greases|lubricant|lubricants|solvent|solvents|detergent|detergents|soap|soaps|shampoo|shampoos|cleaner|cleaners|disinfectant|disinfectants|antiseptic|antiseptics|medicine|drug|drugs|pill|tablet|tablets|capsule|capsules|injection|injections|vaccine|vaccines|antibiotic|antibiotics|painkiller|painkillers|anesthetic|anesthetics|sedative|sedatives|stimulant|stimulants|vitamin|supplement|supplements|herb|herbs|remedy|remedies|cure|treatment|therapy|therapies|surgery|operation|operations|procedure|diagnosis|diagnoses|symptom|symptoms|sign|condition|disease|illness|illnesses|sickness|sicknesses|disorder|disorders|syndrome|syndromes|infection|infections|virus|viruses|bacteria|bacterium|germ|germs|microbe|microbes|parasite|parasites|fungus|fungi|cancer|tumor|tumors|diabetes|asthma|allergy|allergies|arthritis|pneumonia|influenza|flu|cold|colds|fever|fevers|headache|headaches|migraine|migraines|pain|ache|aches|sore|sores|wound|injury|fracture|fractures|sprain|sprains|strain|bruise|bruises|cut|scratch|scratches|burn|burns|bite|bites|sting|stings|rash|rashes|itch|itches|swelling|swellings|inflammation|inflammations|infection|abscess|abscesses|ulcer|ulcers|blister|blisters|cyst|cysts|tumor|cancer|malignancy|malignancies|benign|malignant|metastasis|metastases|remission|remissions|relapse|relapses|recovery|recoveries|rehabilitation|rehabilitations|therapy|treatment|cure|healing|healings|health|wellness|fitness|strength|endurance|stamina|flexibility|balance|coordination|agility|speed|power|energy|vitality|vigor|immunity|resistance|resilience|adaptation|adaptations|evolution|mutation|mutations|variation|inheritance|heredity|gene|genes|chromosome|chromosomes|DNA|RNA|genome|genomes|allele|alleles|trait|phenotype|phenotypes|genotype|genotypes|dominant|recessive|hybrid|hybrids|clone|clones|stem|cell|embryo|embryos|fetus|fetuses|baby|infant|child|adolescent|adolescents|adult|elderly|age|aging|growth|development|maturity|maturities|puberty|menopause|pregnancy|pregnancies|birth|labor|delivery|deliveries|conception|fertilization|fertilizations|ovulation|menstruation|menstruations|reproduction|reproductions|sexuality|sexualities|gender|genders|male|males|female|females|masculine|feminine|heterosexual|homosexual|bisexual|transgender|transsexual|intersex|identity|orientation|orientations|attraction|attractions|desire|desires|arousal|arousals|orgasm|orgasms|intercourse|sex|mating|matings|courtship|courtships|flirtation|flirtations|romance|love|affection|affections|intimacy|intimacies|passion|passions|lust|infatuation|infatuations|crush|crushes|attraction|chemistry|compatibility|compatibilities|commitment|commitments|loyalty|loyalties|fidelity|infidelity|betrayal|betrayals|trust|mistrust|jealousy|jealousies|envy|envies|resentment|resentments|anger|rage|fury|wrath|hatred|hate|spite|malice|vengeance|revenge|retaliation|retaliations|forgiveness|reconciliation|reconciliations|apology|apologies|regret|regrets|remorse|guilt|shame|embarrassment|humiliation|humiliations|pride|vanity|arrogance|humility|modesty|confidence|insecurity|insecurities|fear|anxiety|worry|concern|stress|pressure|tension|nervousness|panic|terror|horror|dread|phobia|phobias|trauma|traumas|shock|surprise|amazement|astonishment|wonder|awe|admiration|respect|reverence|worship|devotion|faith|belief|trust|hope|optimism|pessimism|cynicism|skepticism|doubt|doubts|uncertainty|uncertainties|confusion|perplexity|bewilderment|curiosity|interest|fascination|obsession|obsessions|addiction|addictions|habit|compulsion|compulsions|impulse|impulses|urge|urges|craving|cravings|appetite|appetites|hunger|thirst|desire|need|needs|want|wants|wish|wishes|dream|aspiration|aspirations|ambition|ambitions|goal|objective|purpose|mission|vision|ideal|ideals|value|values|principle|ethic|ethics|moral|morals|virtue|virtues|vice|vices|sin|sins|good|evil|right|wrong|justice|injustice|fairness|unfairness|equality|inequality|freedom|liberty|rights|duty|duties|responsibility|responsibilities|obligation|obligations|law|rule|regulation|code|codes|standard|norm|norms|custom|convention|conventions|protocol|protocols|etiquette|manner|manners|courtesy|courtesies|politeness|rudeness|respect|disrespect|honor|dishonor|dignity|indignity|reputation|reputations|fame|infamy|glory|shame|success|failure|victory|defeat|triumph|disaster|achievement|accomplishment|excellence|mediocrity|perfection|imperfection|flaw|flaws|defect|defects|mistake|error|fault|faults|weakness|strength|advantage|disadvantage|benefit|harm|help|hindrance|aid|obstacle|support|opposition|cooperation|competition|collaboration|conflict|harmony|discord|peace|war|violence|aggression|defense|attack|protection|vulnerability|safety|danger|security|insecurity|risk|caution|recklessness|prudence|wisdom|folly|intelligence|stupidity|genius|idiocy|knowledge|ignorance|education|learning|teaching|training|instruction|guidance|advice|counsel|suggestion|suggestions|recommendation|recommendations|warning|warnings|caution|alert|alerts|alarm|alarms|signal|sign|indication|indications|clue|clues|hint|hints|evidence|proof|proofs|fact|truth|lie|deception|honesty|dishonesty|sincerity|hypocrisy|authenticity|falseness|reality|illusion|appearance|essence|substance|form|content|meaning|significance|importance|relevance|irrelevance|value|worth|merit|quality|quantity|degree|extent|intensity|magnitude|scale|proportion|balance|imbalance|symmetry|asymmetry|order|chaos|organization|disorganization|structure|randomness|pattern|irregularity|consistency|inconsistency|stability|instability|permanence|change|continuity|discontinuity|progression|regression|evolution|revolution|transformation|stagnation|growth|decay|expansion|contraction|increase|decrease|rise|fall|improvement|deterioration|progress|decline|advance|retreat|forward|backward|up|down|high|low|fast|slow|quick|gradual|sudden|smooth|rough|easy|difficult|simple|complex|clear|unclear|obvious|subtle|direct|indirect|straight|curved|flat|steep|level|uneven|smooth|rough|soft|hard|flexible|rigid|elastic|brittle|strong|weak|durable|fragile|solid|liquid|gas|plasma|hot|cold|warm|cool|dry|wet|moist|arid|humid|fresh|stale|new|old|young|ancient|modern|traditional|contemporary|future|past|present|early|late|first|last|beginning|end|start|finish|opening|closing|entrance|exit|arrival|departure|birth|death|creation|destruction|origin|extinction|source|destination|cause|effect|reason|result|purpose|accident|intention|plan|chance|fate|destiny|fortune|luck|probability|certainty|possibility|impossibility|likelihood|unlikelihood|common|rare|ordinary|extraordinary|normal|abnormal|typical|unusual|regular|irregular|frequent|infrequent|constant|variable|uniform|diverse|similar|different|same|other|equal|unequal|identical|unique|general|specific|universal|particular|whole|part|all|none|some|many|few|most|least|more|less|much|little|enough|insufficient|excess|lack|abundance|scarcity|surplus|deficit|profit|loss|gain|cost|price|value|cheap|expensive|free|costly|rich|poor|wealth|poverty|luxury|necessity|comfort|hardship|ease|difficulty|pleasure|pain|joy|sorrow|happiness|sadness|laughter|tears|smile|frown|fun|boredom|excitement|calm|enthusiasm|apathy|energy|fatigue|strength|weakness|health|sickness|life|death|youth|age|beauty|ugliness|grace|clumsiness|elegance|vulgarity|refinement|coarseness|sophistication|simplicity|complexity|clarity|confusion|order|chaos|peace|conflict|love|hate|friend|enemy|ally|opponent|supporter|critic|advocate|adversary|partner|rival|colleague|competitor|companion|stranger|neighbor|foreigner|native|citizen|alien|resident|visitor|host|guest|leader|follower|master|servant|teacher|student|parent|child|husband|wife|brother|sister|son|daughter|father|mother|grandfather|grandmother|uncle|aunt|nephew|niece|cousin|relative|ancestor|descendant|family|stranger|friend|acquaintance|lover|spouse|partner|mate|date|boyfriend|girlfriend|fiance|fiancee|bride|groom|widow|widower|orphan|adoptee|guardian|ward|mentor|protege|coach|trainee|boss|employee|employer|worker|manager|subordinate|executive|staff|owner|tenant|landlord|renter|buyer|seller|customer|vendor|client|supplier|patient|doctor|nurse|therapist|counselor|advisor|consultant|expert|amateur|professional|specialist|generalist|novice|veteran|beginner|master|apprentice|journeyman|craftsman|artisan|artist|performer|athlete|player|competitor|champion|winner|loser|hero|villain|victim|survivor|witness|participant|observer|spectator|audience|crowd|individual|group|team|organization|institution|company|corporation|business|firm|agency|department|division|branch|office|bureau|committee|board|council|assembly|congress|parliament|senate|court|tribunal|jury|judge|lawyer|attorney|prosecutor|defendant|plaintiff|witness|bailiff|clerk|reporter|editor|journalist|writer|author|poet|novelist|playwright|screenwriter|director|producer|actor|actress|singer|dancer|musician|composer|conductor|painter|sculptor|photographer|designer|architect|engineer|builder|contractor|carpenter|plumber|electrician|mechanic|technician|repairman|installer|operator|driver|pilot|captain|sailor|soldier|officer|general|admiral|sergeant|private|recruit|veteran|civilian|police|detective|sheriff|deputy|guard|watchman|patrol|inspector|investigator|agent|spy|informant|criminal|thief|robber|burglar|murderer|killer|assassin|terrorist|rebel|revolutionary|activist|protester|demonstrator|rioter|vandal|arsonist|smuggler|dealer|trafficker|pirate|hijacker|kidnapper|rapist|molester|abuser|bully|harasser|stalker|blackmailer|extortionist|forger|counterfeiter|embezzler|swindler|fraud|cheat|liar|deceiver|impostor|pretender|hypocrite|traitor|betrayer|turncoat|defector|deserter|coward|hero|martyr|saint|angel|devil|demon|monster|ghost|spirit|soul|deity|god|goddess|creator|destroyer|savior|messiah|prophet|priest|minister|pastor|rabbi|imam|monk|nun|missionary|pilgrim|believer|worshiper|follower|disciple|apostle|convert|heretic|infidel|atheist|agnostic|skeptic|fanatic|zealot|extremist|moderate|liberal|conservative|radical|reactionary|progressive|traditionalist|reformer|revolutionary|loyalist|patriot|nationalist|internationalist|globalist|isolationist|interventionist|pacifist|militarist|hawk|dove|left|right|center|middle|extreme|mainstream|fringe|establishment|opposition|majority|minority|plurality|consensus|dissent|agreement|disagreement|unity|division|cooperation|competition|collaboration|confrontation|negotiation|compromise|concession|demand|offer|proposal|suggestion|recommendation|request|appeal|plea|petition|protest|complaint|objection|criticism|praise|approval|disapproval|support|opposition|endorsement|rejection|acceptance|refusal|consent|denial|permission|prohibition|allowance|restriction|freedom|constraint|liberty|bondage|independence|dependence|autonomy|control|power|weakness|authority|submission|dominance|subordination|superiority|inferiority|equality|hierarchy|rank|status|position|role|function|duty|right|privilege|entitlement|claim|obligation|debt|credit|asset|liability|resource|burden|advantage|handicap|strength|vulnerability|opportunity|threat|potential|limitation|possibility|impossibility|capability|incapacity|ability|disability|talent|deficiency|skill|incompetence|knowledge|ignorance|wisdom|foolishness|experience|inexperience|expertise|amateurism|mastery|failure|success|achievement|defeat|victory|loss|gain|profit|benefit|harm|help|hindrance|aid|obstacle|support|resistance|cooperation|opposition|agreement|conflict|harmony|discord|peace|war|love|hate|friendship|enmity|trust|suspicion|faith|doubt|hope|despair|joy|sorrow|pleasure|pain|comfort|discomfort|ease|difficulty|simplicity|complexity|clarity|confusion|certainty|uncertainty|truth|falsehood|reality|illusion|fact|fiction|actual|potential|concrete|abstract|specific|general|particular|universal|individual|collective|personal|impersonal|subjective|objective|relative|absolute|temporary|permanent|transient|eternal|finite|infinite|limited|unlimited|bounded|boundless|measurable|immeasurable|visible|invisible|tangible|intangible|material|spiritual|physical|mental|bodily|intellectual|emotional|rational|conscious|unconscious|voluntary|involuntary|intentional|accidental|deliberate|spontaneous|planned|unplanned|organized|chaotic|systematic|random|methodical|haphazard|careful|careless|precise|vague|exact|approximate|accurate|inaccurate|correct|incorrect|right|wrong|true|false|valid|invalid|sound|unsound|logical|illogical|reasonable|unreasonable|rational|irrational|sane|insane|normal|abnormal|healthy|sick|well|ill|fit|unfit|able|unable|capable|incapable|competent|incompetent|qualified|unqualified|suitable|unsuitable|appropriate|inappropriate|proper|improper|decent|indecent|moral|immoral|ethical|unethical|legal|illegal|lawful|unlawful|just|unjust|fair|unfair|equal|unequal|balanced|imbalanced|neutral|biased|objective|subjective|impartial|partial|disinterested|interested|detached|involved|distant|close|remote|near|far|high|low|deep|shallow|wide|narrow|broad|thin|thick|heavy|light|dense|sparse|full|empty|complete|incomplete|whole|partial|entire|fractional|total|limited|comprehensive|exclusive|inclusive|pure|mixed|clean|dirty|clear|muddy|transparent|opaque|bright|dark|light|shadow|white|black|colored|colorless|vivid|dull|shiny|matte|glossy|flat|smooth|rough|soft|hard|tender|tough|flexible|rigid|elastic|stiff|fluid|solid|liquid|gas|wet|dry|moist|arid|hot|cold|warm|cool|burning|freezing|boiling|frozen|melting|evaporating|condensing|expanding|contracting|growing|shrinking|increasing|decreasing|rising|falling|ascending|descending|climbing|dropping|soaring|plummeting|floating|sinking|flying|grounding|moving|stationary|traveling|staying|going|coming|leaving|arriving|departing|entering|exiting|approaching|receding|advancing|retreating|progressing|regressing|proceeding|stopping|continuing|pausing|resuming|starting|finishing|beginning|ending|opening|closing|connecting|disconnecting|joining|separating|uniting|dividing|merging|splitting|combining|isolating|integrating|segregating|including|excluding|adding|subtracting|multiplying|dividing|increasing|reducing|expanding|limiting|extending|shortening|lengthening|widening|narrowing|deepening|shallowing|heightening|lowering|strengthening|weakening|hardening|softening|tightening|loosening|fastening|unfastening|securing|releasing|holding|dropping|grasping|letting|catching|throwing|lifting|lowering|raising|dropping|carrying|setting|placing|removing|inserting|extracting|pushing|pulling|pressing|releasing|squeezing|relaxing|stretching|compressing|bending|straightening|folding|unfolding|rolling|unrolling|wrapping|unwrapping|tying|untying|binding|freeing|attaching|detaching|sticking|peeling|gluing|separating|welding|cutting|sewing|tearing|mending|breaking|repairing|fixing|damaging|building|destroying|creating|eliminating|making|unmaking|forming|deforming|shaping|reshaping|molding|casting|carving|sculpting|painting|erasing|drawing|sketching|coloring|shading|highlighting|outlining|filling|emptying|loading|unloading|packing|unpacking|storing|retrieving|saving|deleting|recording|erasing|writing|reading|typing|printing|copying|pasting|cutting|editing|formatting|styling|decorating|plain|designing|planning|organizing|arranging|ordering|sorting|classifying|categorizing|grouping|separating|mixing|blending|combining|dividing|distributing|collecting|gathering|scattering|spreading|concentrating|diluting|purifying|contaminating|cleaning|dirtying|washing|drying|polishing|tarnishing|shining|dulling|sharpening|blunting|smoothing|roughening|leveling|tilting|balancing|unbalancing|stabilizing|destabilizing|securing|endangering|protecting|exposing|covering|uncovering|hiding|revealing|concealing|displaying|showing|demonstrating|illustrating|explaining|confusing|clarifying|complicating|simplifying|elaborating|summarizing|detailing|generalizing|specifying|broadening|focusing|concentrating|dispersing|centralizing|decentralizing|unifying|fragmenting|consolidating|splitting|strengthening|undermining|supporting|opposing|promoting|hindering|encouraging|discouraging|enabling|preventing|allowing|forbidding|permitting|prohibiting|authorizing|banning|approving|rejecting|endorsing|condemning|praising|criticizing|commending|blaming|rewarding|punishing|compensating|penalizing|crediting|debiting|paying|charging|buying|selling|trading|exchanging|giving|taking|lending|borrowing|investing|divesting|saving|spending|earning|losing|gaining|forfeiting|winning|surrendering|acquiring|disposing|keeping|discarding|retaining|releasing|maintaining|neglecting|preserving|destroying|conserving|wasting|using|saving|consuming|producing|recycling|disposing|hoarding|sharing|monopolizing|distributing|concentrating|spreading|localizing|globalizing|importing|exporting|shipping|receiving|sending|delivering|returning|forwarding|redirecting|broadcasting|narrowcasting|publishing|suppressing|announcing|concealing|declaring|denying|confirming|refuting|asserting|questioning|stating|implying|suggesting|hinting|expressing|repressing|communicating|withholding|informing|misinforming|educating|misleading|enlightening|confusing|teaching|learning|instructing|following|guiding|misguiding|directing|misdirecting|leading|following|commanding|obeying|ordering|complying|demanding|yielding|insisting|conceding|requiring|waiving|mandating|volunteering|forcing|allowing|compelling|permitting|obliging|excusing|binding|releasing|committing|absolving|promising|breaking|pledging|reneging|vowing|betraying|swearing|forswearing|guaranteeing|defaulting|ensuring|risking|securing|jeopardizing|protecting|endangering|defending|attacking|guarding|exposing|shielding|revealing|fortifying|weakening|reinforcing|undermining|arming|disarming|equipping|stripping|supplying|depriving|providing|withholding|furnishing|removing|stocking|depleting|filling|emptying|loading|unloading|charging|discharging|fueling|draining|powering|exhausting|energizing|tiring|activating|deactivating|starting|stopping|initiating|terminating|launching|aborting|triggering|preventing|causing|avoiding|producing|eliminating|generating|destroying|creating|annihilating|originating|ending|inventing|copying|innovating|imitating|pioneering|following|discovering|hiding|uncovering|concealing|exploring|avoiding|investigating|ignoring|researching|neglecting|studying|overlooking|analyzing|disregarding|examining|dismissing|inspecting|neglecting|scrutinizing|ignoring|observing|overlooking|watching|ignoring|monitoring|neglecting|tracking|losing|following|abandoning|pursuing|escaping|chasing|evading|hunting|hiding|seeking|avoiding|searching|ignoring)\b/gi,
            nouns: /\b[A-Z][a-z]+\b/g, // Capitalized words as potential proper nouns
            questions: /\b(who|what|when|where|why|how|which|whom|whose)\b/gi,
            negations: /\b(not|no|never|none|nothing|nobody|nowhere|neither|nor)\b/gi,
            conjunctions: /\b(and|or|but|yet|so|for|nor|as|if|then|because|since|unless|although|though|while|whereas|whether)\b/gi,
            prepositions: /\b(in|on|at|to|for|from|with|by|about|through|during|before|after|above|below|between|under|over|among|within|without|toward|against|across|behind|beyond|inside|outside|beneath|beside|besides|along|around|upon|unto|into|onto|throughout|underneath)\b/gi,
            pronouns: /\b(I|me|my|mine|myself|you|your|yours|yourself|he|him|his|himself|she|her|hers|herself|it|its|itself|we|us|our|ours|ourselves|they|them|their|theirs|themselves|this|that|these|those|who|whom|whose|which|what|whoever|whomever|whatever|whichever|one|ones|all|some|any|each|every|either|neither|both|few|many|several|all|most|none)\b/gi,
            articles: /\b(a|an|the)\b/gi,
            numbers: /\b(\d+)\b/g,
            punctuation: /([.!?;:,])/g
        };

        let highlighted = this._escapeHtml(text);
        
        // Apply highlighting in specific order to avoid conflicts
        highlighted = highlighted.replace(patterns.numbers, '<span class="syntax-number">$&</span>');
        highlighted = highlighted.replace(patterns.questions, '<span class="syntax-question">$&</span>');
        highlighted = highlighted.replace(patterns.negations, '<span class="syntax-negation">$&</span>');
        highlighted = highlighted.replace(patterns.adverbs, '<span class="syntax-adverb">$&</span>');
        highlighted = highlighted.replace(patterns.adjectives, '<span class="syntax-adjective">$&</span>');
        highlighted = highlighted.replace(patterns.verbs, '<span class="syntax-verb">$&</span>');
        highlighted = highlighted.replace(patterns.pronouns, '<span class="syntax-pronoun">$&</span>');
        highlighted = highlighted.replace(patterns.prepositions, '<span class="syntax-preposition">$&</span>');
        highlighted = highlighted.replace(patterns.conjunctions, '<span class="syntax-conjunction">$&</span>');
        highlighted = highlighted.replace(patterns.articles, '<span class="syntax-article">$&</span>');
        highlighted = highlighted.replace(patterns.nouns, '<span class="syntax-proper-noun">$&</span>');
        
        return highlighted;
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    applySyntaxHighlighting(html) {
        if (!this.syntaxHighlightingEnabled) return html;
        
        try {
            const temp = document.createElement('div');
            temp.innerHTML = html;
            
            // Find all text nodes and apply highlighting
            const walker = document.createTreeWalker(
                temp,
                NodeFilter.SHOW_TEXT,
                {
                    acceptNode: function(node) {
                        const parent = node.parentElement;
                        if (parent.matches('code, pre, .syntax-code, .syntax-code-block, [class^="syntax-"]')) {
                            return NodeFilter.FILTER_REJECT;
                        }
                        return NodeFilter.FILTER_ACCEPT;
                    }
                },
                false
            );
            
            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                textNodes.push(node);
            }
            
            textNodes.forEach(textNode => {
                const highlighted = this.highlightNaturalLanguage(textNode.textContent);
                if (highlighted !== textNode.textContent) {
                    const span = document.createElement('span');
                    span.innerHTML = highlighted;
                    textNode.parentNode.replaceChild(span, textNode);
                }
            });
            
            return temp.innerHTML;
        } catch (e) {
            console.error('Error applying syntax highlighting:', e);
            return html;
        }
    }

    toggleBionicMode() {
        this.bionicMode = !this.bionicMode;
        return this.bionicMode;
    }

    toggleFocusGradient() {
        this.focusGradient = !this.focusGradient;
        return this.focusGradient;
    }

    toggleSyntaxHighlighting() {
        this.syntaxHighlightingEnabled = !this.syntaxHighlightingEnabled;
        return this.syntaxHighlightingEnabled;
    }

    formatText(text, options = {}) {
        const {
            isMarkdown = false,
            bionicMode = this.bionicMode,
            syntaxHighlighting = this.syntaxHighlightingEnabled
        } = options;

        let formatted = text;

        if (isMarkdown) {
            formatted = this.parseMarkdown(formatted);
        } else if (bionicMode) {
            formatted = this.applyBionicReading(formatted);
        }

        if (syntaxHighlighting) {
            formatted = this.applySyntaxHighlighting(formatted);
        }

        return formatted;
    }
}