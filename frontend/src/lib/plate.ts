// One plate normaliser for the whole client.
//
// This lived inline in Journey.tsx, and the watchlist add form needed the same
// rule. Two normalisers that agree today drift the first time one of them
// learns about a new separator, and the disagreement only shows up against the
// real API as a 404 on a plate the user can see is correct. So there is exactly
// one, and both callers import it.
//
// It knowingly duplicates backend logic -- D-015 records why that is accepted:
// the path parameter is normalised, and sending the raw string guarantees a
// miss. If the backend normalises differently we will see 404s on plates that
// look right, which is the symptom to remember.
//
// Deliberately NOT shared with src/mocks/handlers.ts. That code is simulating
// the SERVER's normalisation, and a mock that imports the client's rule can
// never disagree with it -- which is precisely the disagreement worth catching.
export function normalisePlate(raw: string): string {
  return raw.toUpperCase().replace(/[^A-Z0-9]/g, '')
}
