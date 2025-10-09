# Phase 1.1 Assessment: Signal Generator Review

**Date:** 2025-10-07
**Status:** Ready for Joint Review

## Summary

Reviewed all 4 signal generators from old_agent code. Code is **functionally reasonable** but requires **complete NumPy→PyTorch conversion** and restructuring to flat src/ layout.

## Detailed Findings by Generator

### 1. GSM Generator (gsm_generator.py)
**Lines of Code:** 129
**Dependencies:** BaseSignalGenerator, config_loader, signal_utils

**Algorithm Assessment:**
- GMSK modulation implementation follows correct approach
- Gaussian filter with BT=0.3 (3GPP TS 45.004 compliant)
- Frequency modulation via phase integration
- **Correctness: GOOD** - matches paper Equation (1)

**Issues Found:**
- Uses NumPy exclusively (np.exp, np.cumsum, np.convolve)
- Import paths assume old structure (from src.signal_generation.base_generator)
- Gaussian filter is simplified approximation (not exact 3GPP filter)
- No emojis in code, but test output may have them

**Conversion Effort:** Medium
- ~200 lines after PyTorch conversion
- Need to handle Gaussian filtering in PyTorch
- torch.cumsum, torch.exp straightforward replacements

### 2. UMTS Generator (umts_generator.py)
**Lines of Code:** 226
**Dependencies:** BaseSignalGenerator, config_loader, modulation (QPSK), signal_utils

**Algorithm Assessment:**
- OVSF spreading code generation correct (tree structure)
- Scrambling code simplified (uses random, not real Gold codes)
- QPSK symbols from shared ModulationSchemes
- Spreading and scrambling operations correct
- **Correctness: FAIR** - spreading is correct, scrambling is simplified

**Issues Found:**
- Uses NumPy throughout
- Scrambling uses np.random with fixed seed (not true Gold codes per 3GPP TS 25.213)
- Pulse shaping is simplified moving average, not root-raised cosine
- Import path dependencies

**Conversion Effort:** Medium-High
- ~300 lines after PyTorch conversion
- OVSF generation needs careful translation
- Consider implementing real Gold codes for authenticity

### 3. LTE Generator (lte_generator.py)
**Lines of Code:** 229
**Dependencies:** BaseSignalGenerator, config_loader, modulation (QAM), signal_utils

**Algorithm Assessment:**
- OFDM implementation follows 3GPP TS 36.211
- FFT size table matches 3GPP specifications
- Cyclic prefix handling correct (varying CP per symbol)
- Subcarrier mapping with guard bands correct
- **Correctness: EXCELLENT** - matches paper Equation (3)

**Issues Found:**
- Uses np.fft.ifft (needs torch.fft.ifft)
- Subcarrier indexing may need verification with PyTorch
- Import dependencies

**Conversion Effort:** Medium
- ~250 lines after PyTorch conversion
- torch.fft.ifft is direct replacement
- FFT scaling factor needs attention (sqrt(fft_size))

### 4. 5G NR Generator (nr_generator.py)
**Lines of Code:** 310
**Dependencies:** BaseSignalGenerator, config_loader, modulation (QAM), signal_utils

**Algorithm Assessment:**
- Flexible numerology implementation correct (μ = 0,1,2,3,4)
- FFT size and RB tables match 3GPP TS 38.211
- Reference signal generation simplified but present
- OFDM generation similar to LTE with numerology flexibility
- **Correctness: EXCELLENT** - matches paper Equation (4)

**Issues Found:**
- Uses NumPy throughout
- Reference signals (DMRS) are simplified, not true 3GPP sequences
- Import dependencies

**Conversion Effort:** Medium-High
- ~350 lines after PyTorch conversion
- Most complex generator due to numerology flexibility
- Reference signal generation needs careful handling

### 5. Base Generator (base_generator.py)
**Lines of Code:** 56
**Abstract base class**

**Assessment:**
- Clean ABC interface with generate_baseband() abstract method
- Utility methods for carrier, noise, power normalization
- Delegates to signal_utils for actual implementations

**Issues Found:**
- Import from ..utils.signal_utils needs fixing
- Needs PyTorch conversion

**Conversion Effort:** Low
- ~70 lines after PyTorch conversion
- Minimal logic, mostly delegates

### 6. Shared Modulation (utils/modulation.py)
**Lines of Code:** 263
**Two classes:** ModulationSchemes, DigitalModulation

**Assessment:**
- QAM constellation generation follows 3GPP specifications
- Supports QPSK, 16QAM, 64QAM, 256QAM, 1024QAM
- Gray coding mentioned but implementation needs verification
- GMSK modulation implementation in DigitalModulation class
- Proper power normalization for all constellations

**Issues Found:**
- All NumPy operations
- modulate_symbols() uses np.packbits (no PyTorch equivalent)

**Conversion Effort:** Medium
- ~300 lines after PyTorch conversion
- Constellation generation straightforward
- modulate_symbols() needs redesign for PyTorch

## Code Quality Assessment

### Strengths
1. **Modular design** - clear separation of concerns
2. **3GPP references** - code mentions relevant specifications
3. **Shared utilities** - no duplication of QAM constellations
4. **Built-in tests** - __main__ blocks for basic validation
5. **Mathematical correctness** - algorithms generally match paper

### Weaknesses
1. **NumPy dependency** - complete framework uses NumPy
2. **Simplified implementations** - several shortcuts taken:
   - UMTS Gold codes are random, not real
   - UMTS pulse shaping is moving average, not RRC
   - 5G DMRS are random phase, not 3GPP sequences
   - GSM Gaussian filter is approximation
3. **No unit tests** - only __main__ test code
4. **Import structure** - assumes nested src/ structure

## Dependencies to Review

These old_agent files need assessment:
1. **signal_utils.py** - normalize_power, add_awgn_noise, add_carrier_frequency, generate_time_vector
2. **config_loader.py** - get_standard_specs() for loading specs from YAML
3. **config/signal_specs.yaml** - Specifications database

## Conversion Strategy

### Approach: **Gradual Rewrite with Validation**

**Step 1: Convert shared utilities first**
- utils_shared.py with PyTorch implementations
- Power normalization, SINR calculation, basic operations

**Step 2: Convert modulation utilities**
- utils_modulation.py with PyTorch constellations
- QAM, QPSK, GMSK functions

**Step 3: Convert generators one by one**
- Start with GSM (simplest)
- Then LTE (most critical for paper)
- Then 5G NR
- Finally UMTS
- Each with validation against NumPy version

**Step 4: Create test suite**
- check/test_gsm.py, test_lte.py, etc.
- Verify output equivalence with old NumPy code
- Check signal properties (power, length, PAPR)

### Device Handling Decision

Add `device='cpu'` parameter to all generators:
```python
class GSMGenerator:
    def __init__(self, ..., device='cpu'):
        self.device = device

    def generate_baseband(self):
        # Generate tensors on self.device
        signal = torch.tensor(..., device=self.device)
```

Rationale:
- Flexibility for future GPU use
- Minimal overhead (just parameter passing)
- Follows PyTorch best practices
- Easy to change default later if needed

## Estimated Work

| Component | NumPy LOC | Est. PyTorch LOC | Effort | Priority |
|-----------|-----------|------------------|--------|----------|
| utils_shared | 150 | 180 | Medium | 1 (foundation) |
| utils_modulation | 263 | 300 | Medium | 2 (needed by all) |
| run_gsm + utils | 185 | 220 | Medium | 3 |
| run_lte + utils | 235 | 270 | Medium | 4 (paper priority) |
| run_5g + utils | 320 | 370 | Medium-High | 5 |
| run_umts + utils | 282 | 340 | Medium-High | 6 |
| **Total** | **~1435** | **~1680** | **2-3 days** | |

## Recommendations

1. **Keep old code architecture** - it's fundamentally sound
2. **Convert to PyTorch systematically** - validate each component
3. **Improve simplifications later** - get working version first, then enhance:
   - Real Gold codes for UMTS
   - Proper RRC filtering
   - Accurate DMRS for 5G
4. **Add comprehensive tests** - pytest suite in check/
5. **Create demo notebooks** - visual verification in check/

## Next Steps (for joint discussion)

1. Review this assessment - agree on conversion approach?
2. Decide conversion order - start with GSM or shared utilities?
3. Define acceptance criteria - how to verify equivalence?
4. Discuss simplifications - fix now or later?

## Files to Migrate

**Must Review:**
- old_agent/src/utils/signal_utils.py
- old_agent/src/utils/config_loader.py
- old_agent/config/signal_specs.yaml

**Can Defer:**
- Channel models (Phase 1.2)
- MIMO (Phase 1.2)
- Signal mixing (Phase 1.3)
